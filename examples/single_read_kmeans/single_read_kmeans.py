#!/usr/bin/env python3
"""
Single-read KMeans pipeline utilities with integrated HDF5 extraction.

Runs every BAM x every region-set combination automatically:
  - each registered BAM -> its own extract at
      {base}/single_reads/extracts/{motif_slug}/{bam_name}/reads.combined_basemods.h5
  - analysis loops over every registered region set ("class") for every BAM.

Register inputs on the CLI (all required; no built-in defaults):
  --base    PATH                (required output/data root)
  --fasta   PATH.fasta          (required single reference)
  --bam     NAME=PATH.bam       (repeatable; at least one required)
  --regions NAME=PATH.bed       (repeatable; at least one required; path used as-is)
  --motifs  "A,0" ["CG,0" ...]  (default: A,0; extracts/results namespaced per motif)

Analysis subcommands (choose what you want):
  paired-plots     -> ONLY the per-cluster pileup figures
  feat-importance  -> ONLY the per-cluster feature-importance reports
  all              -> BOTH paired-plots and feat-importance

HDF5 extraction is automatic by default (missing/corrupt .h5 are (re)built).
Disable with --no-auto-build-h5 to only run on pre-existing h5 files.

Implements:
  (A) QC filtering (before clustering/plotting/FI)
  (B) Feature blocks (autocorr, window densities, run-metrics)
  (C) Read-level metrics (10 criteria; 12 numeric cols; C(r) uses 3 radii)
  (D) Adaptive PCA: n_pca_eff = min(N_PCA, n_samples, n_features)
  (E) Feature scaling (StandardScaler) for equal weighting during clustering.
  (F) Configurable y-axis bounds for the per-cluster pileup (m6A) plots; by
      default reproduces the original auto-scaled behavior.
"""

import os, json, time, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dimelo import load_processed, parse_bam

from kmeans_interp.kmeans_feature_imp import KMeansInterp


# ---------------------------- default params ----------------------------
DEFAULT_BASE = None                          # must be provided via --base
# reads / extraction
MOTIFS = ["A,0"]           # overridable via --motifs
W = 2000
ORIENT = True

# clustering/features
N_PCA = 8
K_MIN, K_MAX = 2, 15
RSTATE = 42

# plotting
SMOOTH_W = 30
PALETTE = plt.get_cmap("tab20").colors

# per-cluster pileup y-bounds (None -> auto, preserving original behavior)
PILEUP_YMIN_DEFAULT = None   # m6A axis lower bound (left y-axis)
PILEUP_YMAX_DEFAULT = None   # m6A axis upper bound (left y-axis)

# ---------------------------- metric params ----------------------------
CENTRAL_RS = (50, 100, 250)
DISTAL_LO, DISTAL_HI = 500, 1000
ENTROPY_K = 40
SHIFT_R = 100
SHIFT_DELTAS = (200, 400, 600, 800)

# ---------------------------- QC filters ----------------------------
MIN_CALLABLE_A = 20
MIN_METHYL_A = 5
MIN_CALLABLE_EACH_SIDE = 5

# ---------------------------- scaling ----------------------------
SCALE_FEATURES_DEFAULT = True  # equal weighting via z-scoring

# ---------------------------- h5 extraction defaults ----------------------------
# No built-in BAMs; register with --bam NAME=PATH.
DEFAULT_BAMS = {}
FASTA = None               # no built-in reference; supply with --fasta
EXTRACT_THRESH = 225
EXTRACT_CORES = 16

# No built-in region sets; register with --regions NAME=PATH.
DEFAULT_CLASSES = {}

# Feature-importance methods
METHODS = [
  ("wcss_min", "wcss_min", False),  # use raw X by default for FI
]


# ---------------------------- small utils ----------------------------
def _motif_slug(motifs):
  """Filesystem-safe tag for a motif list, e.g. ['A,0'] -> 'A-0'."""
  parts = []
  for m in motifs:
    parts.append(str(m).replace(",", "-").replace("/", "_").replace(" ", ""))
  return "_".join(parts)


def _parse_name_path_pairs(items):
  """Parse repeatable 'name=path' CLI args into an ordered dict (or None)."""
  if not items:
    return None
  out = {}
  for it in items:
    if "=" not in it:
      raise ValueError(f"Expected name=path, got: {it!r}")
    name, path = it.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
      raise ValueError(f"Bad name=path entry: {it!r}")
    out[name] = path
  return out


# ---------------------------- config ----------------------------
def make_config(base, bams=None, fasta=None, classes=None):
  """Build a global config. Per-BAM (tag-specific) paths are derived on demand
  via tag_paths(cfg, tag). Extracts/results are namespaced per motif.

  Requires: base, fasta, at least one BAM, and at least one region set.
  All input paths (BAMs, region BEDs, FASTA) are used exactly as given."""
  if not base:
    raise ValueError("No output root provided. Supply one with --base PATH")

  # Region sets: paths used exactly as given.
  if classes is None:
    classes = dict(DEFAULT_CLASSES)  # empty by default
  classes_resolved = dict(classes)
  if not classes_resolved:
    raise ValueError("No region sets provided. Register at least one with "
                     "--regions NAME=PATH.bed")

  bams_resolved = dict(DEFAULT_BAMS) if bams is None else dict(bams)
  if not bams_resolved:
    raise ValueError("No BAMs provided. Register at least one with "
                     "--bam NAME=PATH.bam")

  fasta_resolved = FASTA if fasta is None else fasta
  if not fasta_resolved:
    raise ValueError("No reference FASTA provided. Supply one with --fasta PATH.fasta")

  motif_slug = _motif_slug(MOTIFS)
  extract_dir = os.path.join(base, "single_reads", "extracts", motif_slug)

  return dict(
    base=base,
    classes=classes_resolved,
    bams=bams_resolved,
    fasta=fasta_resolved,
    extract_dir=extract_dir,
    motif_slug=motif_slug,
  )

def tag_paths(cfg, tag):
  """Per-BAM (tag) derived paths: h5, labels_root, fi_root — namespaced by motif."""
  ms = cfg["motif_slug"]
  h5 = os.path.join(cfg["extract_dir"], tag, "reads.combined_basemods.h5")
  labels_root = os.path.join(cfg["base"], "single_reads", "single_reads_kmeans", ms, tag)
  fi_root = os.path.join(cfg["base"], "single_reads", f"kmeans_feature_importance_{ms}", tag)
  return dict(tag=tag, h5=h5, labels_root=labels_root,
              out_paired_root=labels_root, out_fi_root=fi_root)


# ---------------------------- h5 extraction helpers ----------------------------
def _h5_is_readable(path):
  """True iff the HDF5 file exists and opens cleanly (guards truncated files)."""
  if not os.path.exists(path) or os.path.getsize(path) == 0:
    return False
  try:
    with h5py.File(path, "r") as f:
      _ = len(f)
    return True
  except Exception as e:
    print(f"[extract] existing {path} is unreadable ({e!r}); will re-extract", flush=True)
    return False


def build_union_bed(cfg, out_path=None):
  """Concatenate all registered region sets into one 'all sites' bed."""
  if out_path is None:
    out_path = os.path.join(cfg["base"], "single_reads", "all_sites.union.bed")
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, "w") as out:
    for bp in cfg["classes"].values():
      with open(bp) as fh:
        for line in fh:
          if line.strip():
            out.write(line if line.endswith("\n") else line + "\n")
  return out_path


def run_extract(cfg, name, bam, union_bed, cores=EXTRACT_CORES,
                fasta=None, thresh=EXTRACT_THRESH, window_size=W, motifs=None):
  """Extract one BAM -> {extract_dir}/{name}/reads.combined_basemods.h5"""
  if fasta is None:
    fasta = cfg["fasta"]
  if motifs is None:
    motifs = MOTIFS
  extract_dir = cfg["extract_dir"]
  os.makedirs(extract_dir, exist_ok=True)
  h5 = os.path.join(extract_dir, name, "reads.combined_basemods.h5")

  if _h5_is_readable(h5):
    print(f"[extract] skip; reusing {h5}", flush=True)
    return h5

  if os.path.exists(h5):
    print(f"[extract] removing stale/corrupt {h5}", flush=True)
    os.remove(h5)
    for stale in glob.glob(os.path.join(os.path.dirname(h5), "reads.*.txt")):
      print(f"[extract] removing stale staging file {stale}", flush=True)
      os.remove(stale)

  if not os.path.exists(bam):
    raise FileNotFoundError(f"BAM not found: {bam}")

  print(f"[extract] {name}: {bam}  motifs={motifs}", flush=True)
  parse_bam.extract(
    input_file=bam,
    output_name=name,
    ref_genome=fasta,
    output_directory=extract_dir,
    regions=str(union_bed),
    motifs=motifs,
    thresh=thresh,
    window_size=window_size,
    cores=cores,
    override_checks=True,
  )
  if not os.path.exists(h5):
    raise FileNotFoundError(f"extract finished but no .h5 at {h5}")
  return h5


def cmd_build_h5(cfg, names=None, cores=EXTRACT_CORES):
  """Build .h5 for one or more (default: all) registered BAMs."""
  for k, p in cfg["classes"].items():
    if not os.path.exists(p):
      raise FileNotFoundError(f"Missing region set {k}: {p}")

  union_bed = build_union_bed(cfg)
  with open(union_bed) as fh:
    n_regions = sum(1 for _ in fh)
  print(f"[union] {union_bed} ({n_regions} regions)", flush=True)

  bams = cfg["bams"]
  if names is None:
    names = list(bams.keys())

  out = {}
  for name in names:
    if name not in bams:
      raise KeyError(f"Unknown BAM name '{name}'. Known: {list(bams.keys())}")
    h5 = run_extract(cfg, name, bams[name], union_bed, cores=cores)
    print(f"[done] {name} -> {h5}", flush=True)
    out[name] = h5
  return out


def ensure_h5(cfg, tag, cores=EXTRACT_CORES):
  """Ensure the h5 for `tag` exists/readable; build it if not."""
  tp = tag_paths(cfg, tag)
  if _h5_is_readable(tp["h5"]):
    return tp["h5"]
  bams = cfg["bams"]
  if tag not in bams:
    raise FileNotFoundError(
      f"h5 not found at {tp['h5']} and tag '{tag}' is not a registered BAM "
      f"(known: {list(bams.keys())}). Register it with --bam {tag}=<path>."
    )
  print(f"[auto-build-h5] {tp['h5']} missing; extracting tag={tag}", flush=True)
  cmd_build_h5(cfg, names=[tag], cores=cores)
  return tp["h5"]


# ---------------------------- shared helpers ----------------------------
def coerce_strand(x):
  if isinstance(x, str):
    x = x.strip()
    if x in {"+", "-"}:
      return x
  return None


def win_from_tuple(t, idx, Wlen, flip, vec_field):
  rs, re_ = t[idx["read_start"]], t[idx["read_end"]]
  v = np.asarray(t[idx[vec_field]], dtype=np.uint8)
  rgs, rge = t[idx["region_start"]], t[idx["region_end"]]

  if not (isinstance(rs, (int, np.integer)) and isinstance(re_, (int, np.integer))):
    return None
  if re_ <= rs:
    return None
  if not (isinstance(rgs, (int, np.integer)) and isinstance(rge, (int, np.integer))):
    return None

  c = (rgs + rge) // 2
  h = Wlen // 2
  ws, we = c - h, c + h - 1
  if ws < rs or we > re_:
    return None

  s, e = ws - rs, we - rs + 1
  if s < 0 or e > v.shape[0]:
    return None

  seg = v[s:e]
  if seg.shape[0] != Wlen:
    return None

  if flip:
    r = coerce_strand(t[idx["region_strand"]])
    if r == "-":
      seg = seg[::-1]
  return seg


def _smooth(v, w=30):
  v = np.asarray(v, dtype=float)
  if w <= 1 or v.size <= 1:
    return v
  pad = w // 2
  vp = np.pad(v, (pad, w - 1 - pad), mode="reflect")
  kernel = np.ones(w, dtype=float) / w
  return np.convolve(vp, kernel, mode="valid")


def _is_dark(rgb):
  r, g, b = rgb[0], rgb[1], rgb[2]
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5


def autocorr(v, lag):
  v = np.asarray(v, float)
  if v.size < lag + 1:
    return 0.0
  x = v - v.mean()
  den = np.dot(x, x)
  if den == 0:
    return 0.0
  return float(np.dot(x[:-lag], x[lag:]) / den)


def _safe_div(num, den, default=np.nan):
  den = float(den)
  if den == 0.0:
    return default
  return float(num) / den


# ---------------------------- QC filtering ----------------------------
def filter_reads_qc(M, V,
                    min_callable=MIN_CALLABLE_A,
                    min_methyl=MIN_METHYL_A,
                    min_callable_each_side=MIN_CALLABLE_EACH_SIDE):
  M = np.asarray(M)
  V = np.asarray(V)
  if M.shape != V.shape:
    raise ValueError(f"filter_reads_qc: shape mismatch M{M.shape} vs V{V.shape}")

  if M.size == 0:
    keep = np.zeros((0,), dtype=bool)
    return M, V, np.where(keep)[0]

  Wlen = M.shape[1]
  c = Wlen // 2

  v = (V > 0).astype(np.uint8)
  m = ((M > 0) & (V > 0)).astype(np.uint8)

  callable_total = v.sum(axis=1)
  methyl_total   = m.sum(axis=1)
  callable_left  = v[:, :c].sum(axis=1)
  callable_right = v[:, c:].sum(axis=1)

  keep = (
    (callable_total >= int(min_callable)) &
    (methyl_total   >= int(min_methyl)) &
    (callable_left  >= int(min_callable_each_side)) &
    (callable_right >= int(min_callable_each_side))
  )
  keep_idx = np.where(keep)[0]
  return M[keep], V[keep], keep_idx


# ---------------------------- per-read metrics ----------------------------
def per_read_metrics(M, V,
                     central_rs=CENTRAL_RS,
                     distal_lo=DISTAL_LO, distal_hi=DISTAL_HI,
                     entropy_k=ENTROPY_K,
                     shift_deltas=SHIFT_DELTAS,
                     shift_r=SHIFT_R):
  M = np.asarray(M)
  V = np.asarray(V)
  if M.shape != V.shape:
    raise ValueError(f"per_read_metrics: shape mismatch M{M.shape} vs V{V.shape}")

  N, Wlen = M.shape
  x = np.arange(-Wlen // 2, Wlen // 2, dtype=float)

  m = ((M > 0) & (V > 0)).astype(np.uint8)
  v = (V > 0).astype(np.uint8)

  central_masks = {int(r): (np.abs(x) <= float(r)) for r in central_rs}
  distal_mask = (np.abs(x) >= float(distal_lo)) & (np.abs(x) <= float(distal_hi))

  B = m.sum(axis=1).astype(float)
  Vtot = v.sum(axis=1).astype(float)
  D = np.array([_safe_div(Bi, Vi, default=np.nan) for Bi, Vi in zip(B, Vtot)], dtype=float)

  Ccols = []
  for r in central_rs:
    mr = m[:, central_masks[int(r)]].sum(axis=1).astype(float)
    Ccols.append(np.array([_safe_div(a, b, default=np.nan) for a, b in zip(mr, B)], dtype=float))
  C = np.stack(Ccols, axis=1)

  r_delta = 250
  if r_delta not in central_masks:
    central_masks[r_delta] = (np.abs(x) <= float(r_delta))

  m_center = m[:, central_masks[r_delta]].sum(axis=1).astype(float)
  v_center = v[:, central_masks[r_delta]].sum(axis=1).astype(float)
  Dcenter = np.array([_safe_div(a, b, default=np.nan) for a, b in zip(m_center, v_center)], dtype=float)

  m_dist = m[:, distal_mask].sum(axis=1).astype(float)
  v_dist = v[:, distal_mask].sum(axis=1).astype(float)
  Ddist = np.array([_safe_div(a, b, default=np.nan) for a, b in zip(m_dist, v_dist)], dtype=float)

  Delta_250 = Dcenter - Ddist

  cm = np.full(N, np.nan, dtype=float)
  R  = np.full(N, np.nan, dtype=float)
  for i in range(N):
    Bi = B[i]
    if Bi <= 0:
      continue
    wi = m[i].astype(float)
    cmi = (x * wi).sum() / Bi
    cm[i] = cmi
    R[i] = np.sqrt(((wi * (x - cmi) ** 2).sum()) / Bi)

  R50 = np.full(N, np.nan, dtype=float)
  R80 = np.full(N, np.nan, dtype=float)
  absx = np.abs(x)
  order = np.argsort(absx)
  absx_sorted = absx[order]
  for i in range(N):
    Bi = B[i]
    if Bi <= 0:
      continue
    mi_sorted = m[i, order].astype(float)
    cum = np.cumsum(mi_sorted) / Bi
    i50 = np.searchsorted(cum, 0.50, side="left")
    i80 = np.searchsorted(cum, 0.80, side="left")
    if i50 < cum.size:
      R50[i] = absx_sorted[i50]
    if i80 < cum.size:
      R80[i] = absx_sorted[i80]

  Hnorm = np.full(N, np.nan, dtype=float)
  K = int(entropy_k)
  edges = np.linspace(0, Wlen, K + 1).astype(int)
  for i in range(N):
    Bi = B[i]
    if Bi <= 0:
      continue
    counts = np.zeros(K, dtype=float)
    for b in range(K):
      s, e = int(edges[b]), int(edges[b + 1])
      counts[b] = m[i, s:e].sum()
    p = counts / Bi
    p = p[p > 0]
    H = -(p * np.log(p)).sum() if p.size else 0.0
    Hnorm[i] = H / np.log(float(K))

  dmin = np.full(N, np.nan, dtype=float)
  for i in range(N):
    pos = np.where(m[i] > 0)[0]
    if pos.size == 0:
      continue
    dmin[i] = float(np.min(np.abs(x[pos])))

  S100 = np.full(N, np.nan, dtype=float)
  true_mask = (np.abs(x) <= float(shift_r))
  for i in range(N):
    Bi = B[i]
    if Bi <= 0:
      continue
    Ctrue = _safe_div(m[i, true_mask].sum(), Bi, default=np.nan)

    shift_vals = []
    for d in shift_deltas:
      for sign in (-1, +1):
        delta = sign * int(d)
        mask_shift = (np.abs(x - float(delta)) <= float(shift_r))
        shift_vals.append(_safe_div(m[i, mask_shift].sum(), Bi, default=np.nan))

    shift_vals = np.asarray(shift_vals, dtype=float)
    shift_vals = shift_vals[~np.isnan(shift_vals)]
    if shift_vals.size == 0:
      continue
    S100[i] = float(Ctrue - np.median(shift_vals))

  out = np.column_stack([
    B, D,
    C,                # C_50,C_100,C_250
    Delta_250,
    cm, R,
    R50, R80,
    Hnorm,
    dmin,
    S100
  ]).astype(float)
  return out


# ---------------------------- robust run features ----------------------------
def _run_features_one(mask, x_bp):
  m = np.asarray(mask, dtype=np.uint8).reshape(-1)
  x_bp = np.asarray(x_bp, dtype=float).reshape(-1)

  Wlen = m.size
  if Wlen == 0:
    return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0

  one_idx = np.flatnonzero(m)
  if one_idx.size == 0:
    return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0

  breaks = np.where(np.diff(one_idx) != 1)[0]
  run_starts = np.r_[0, breaks + 1]
  run_ends = np.r_[breaks, one_idx.size - 1]

  n_runs = int(run_starts.size)
  lengths = (run_ends - run_starts + 1).astype(int)

  max_i = int(np.argmax(lengths))
  max_len = int(lengths[max_i])

  sL = int(one_idx[run_starts[max_i]])
  eL = int(one_idx[run_ends[max_i]])
  center_idx = (sL + eL) / 2.0

  denom = max(1.0, (Wlen - 1))
  center_norm = center_idx / denom

  mid = (Wlen - 1) / 2.0
  edge_norm = abs(center_idx - mid) / max(1.0, mid)

  run_vals = np.zeros(n_runs, dtype=float)
  for r in range(n_runs):
    s = int(one_idx[run_starts[r]])
    e = int(one_idx[run_ends[r]])
    L = int(lengths[r])

    c_idx = (s + e) / 2.0
    lo = int(np.floor(c_idx))
    hi = int(np.ceil(c_idx))
    if hi == lo:
      c_bp = x_bp[lo]
    else:
      frac = c_idx - lo
      c_bp = (1 - frac) * x_bp[lo] + frac * x_bp[hi]

    run_vals[r] = L * float(c_bp)

  pos_wsum = float(run_vals.sum())
  pos_maxabs = float(run_vals[int(np.argmax(np.abs(run_vals)))])

  return float(n_runs), float(max_len), float(center_norm), float(edge_norm), pos_wsum, pos_maxabs


def run_features(M, V):
  M = np.asarray(M)
  V = np.asarray(V)
  if M.shape != V.shape:
    raise ValueError(f"run_features: shape mismatch M{M.shape} vs V{V.shape}")
  N, Wlen = M.shape
  x_bp = np.arange(-Wlen // 2, Wlen // 2, dtype=float)
  mask = ((V > 0) & (M > 0)).astype(np.uint8)

  out = np.zeros((N, 6), dtype=float)
  for i in range(N):
    out[i, :] = _run_features_one(mask[i], x_bp)
  return out


# ---------------------------- featurization (adaptive PCA) ----------------------------
def featurize(M, V=None, n_pca=N_PCA, rstate=RSTATE):
  M = np.asarray(M)
  N, Wlen = M.shape

  lags = [67, 70, 93, 167, 210]
  ac = np.stack([[autocorr(row, L) for L in lags] for row in M])

  c = Wlen // 2
  dens = np.stack([
    M[:, c-25:c+25].mean(1),
    M[:, c-37:c+38].mean(1),
    M[:, c-50:c+50].mean(1),
    M[:, c-150:c+150].mean(1),
    M[:, c-250:c+250].mean(1),
  ], axis=1)

  n_pca_eff = int(min(int(n_pca), int(N), int(Wlen)))
  if n_pca_eff >= 1:
    pca = PCA(n_components=n_pca_eff, random_state=rstate)
    pcs = pca.fit_transform(M.astype(float))
  else:
    pca = None
    pcs = np.zeros((N, 0), dtype=float)

  parts = [pcs, ac, dens]
  if V is not None and V.shape == M.shape:
    parts.append(run_features(M, V))
    parts.append(per_read_metrics(M, V))

  X = np.hstack(parts).astype(float)
  return X, pca, n_pca_eff


def build_feature_names(n_pca_eff,
                        central_rs=CENTRAL_RS,
                        r_delta=250,
                        shift_r=SHIFT_R):
  names = []
  names += [f"PC{i+1}" for i in range(int(n_pca_eff))]
  names += ["ac_lag67", "ac_lag70", "ac_lag93", "ac_lag167", "ac_lag210"]
  names += ["dens_w50", "dens_w75", "dens_w100", "dens_w300", "dens_w500"]
  names += [
    "modA_run_n",
    "modA_run_maxlen",
    "modA_run_long_center",
    "modA_run_long_edge",
    "modA_run_pos_wsum",
    "modA_run_pos_maxabs",
  ]
  names += ["B", "D"]
  names += [f"C_{int(r)}" for r in central_rs]
  names += [f"Delta_{int(r_delta)}", "cm", "R", "R50", "R80", "Hnorm", "dmin", f"S_{int(shift_r)}"]
  return tuple(names)


# ---------------------------- load reads (M,V) ----------------------------
def load_class_mod_val(h5, bed):
  """Load paired windows: mod_vector (M) and val_vector (V), row-aligned."""
  t0 = time.time()
  reads, fields, _ = load_processed.read_vectors_from_hdf5(
    file=h5, motifs=MOTIFS, regions=bed, window_size=W, span_full_window=True
  )
  idx = {f: i for i, f in enumerate(fields)}

  rows_mod, rows_val = [], []
  for t in reads:
    wm = win_from_tuple(t, idx, W, ORIENT, "mod_vector")
    if wm is None:
      continue
    wv = win_from_tuple(t, idx, W, ORIENT, "val_vector")
    if wv is None:
      wv = np.zeros(W, dtype=np.uint8)
    rows_mod.append(wm)
    rows_val.append(wv)

  if rows_mod:
    M = np.vstack(rows_mod).astype(np.uint8)
    V = np.vstack(rows_val).astype(np.uint8)
  else:
    M = np.zeros((0, W), np.uint8)
    V = np.zeros((0, W), np.uint8)

  print(f"  loaded {len(reads)} tuples -> {M.shape[0]} full-window reads in {time.time()-t0:.1f}s",
        flush=True)
  return M, V


# ---------------------------- label caching with scaling ----------------------------
def labels_basepath(tp, cls, k):
  return os.path.join(tp["labels_root"], cls, f"k{k:02d}_labels")


def load_or_fit_labels_scaled(X, tp, cls, k, scale_features=True):
  base = labels_basepath(tp, cls, k)
  suffix = ".scaled.npz" if scale_features else ".raw.npz"
  p = base + suffix

  if os.path.exists(p):
    z = np.load(p, allow_pickle=False)
    labels = z["labels"].astype(int)
    if labels.shape[0] == X.shape[0]:
      if scale_features:
        scaler = StandardScaler()
        scaler.mean_ = z["scaler_mean"].astype(float)
        scaler.scale_ = z["scaler_scale"].astype(float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        return labels, True, scaler
      return labels, True, None

  if scale_features:
    scaler = StandardScaler().fit(X)
    Xk = scaler.transform(X)
  else:
    scaler = None
    Xk = X

  km = KMeans(n_clusters=k, n_init=10, random_state=RSTATE)
  raw = km.fit_predict(Xk)

  from collections import Counter
  counts = Counter(raw)
  order = sorted(counts.items(), key=lambda z: z[1], reverse=True)
  remap = {old: new for new, (old, _) in enumerate(order)}
  labels = np.array([remap[c] for c in raw], dtype=int)

  os.makedirs(os.path.dirname(p), exist_ok=True)
  if scale_features:
    np.savez(
      p,
      labels=labels.astype(np.int16),
      scaler_mean=scaler.mean_.astype(np.float64),
      scaler_scale=scaler.scale_.astype(np.float64),
    )
  else:
    np.savez(p, labels=labels.astype(np.int16))

  return labels, False, scaler


# ---------------------------- paired plots ----------------------------
def make_paired_figure_replot_style(M, V, labels, cls, k, outdir, tag,
                                     pileup_ymin=PILEUP_YMIN_DEFAULT,
                                     pileup_ymax=PILEUP_YMAX_DEFAULT):
  """
  Per-cluster pileup y-bounds:
    pileup_ymin / pileup_ymax control the left (m6A) axis limits of the
    per-cluster profile panels. Either may be None:
      - pileup_ymin is None -> 0.0 (original behavior)
      - pileup_ymax is None -> auto (original 1.15x auto-scaling behavior)
  """
  N, Wlen = M.shape
  ks = sorted(np.unique(labels))
  colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(ks)}
  x = np.arange(-Wlen // 2, Wlen // 2)

  MAX_PER_CLUSTER = 1500
  MAX_TOTAL = 6000
  raw_counts = {c: int(np.sum(labels == c)) for c in ks}
  capped = {c: min(v, MAX_PER_CLUSTER) for c, v in raw_counts.items()}
  tot = sum(capped.values())
  if tot > MAX_TOTAL:
    scale = MAX_TOTAL / tot
    capped = {c: max(20, int(v * scale)) for c, v in capped.items()}

  scat_x = []; scat_y = []; scat_c = []
  y_edges = [0]
  y0 = 0
  for c in ks:
    sel = np.where(labels == c)[0]
    order = np.argsort(-M[sel].mean(1))
    sub = M[sel[order]]
    n_target = capped[c]
    if sub.shape[0] > n_target:
      idx_keep = np.linspace(0, sub.shape[0] - 1, n_target).astype(int)
      sub = sub[idx_keep]
    h = sub.shape[0]
    ry, rj = np.where(sub > 0)
    if ry.size:
      scat_x.append(x[rj])
      scat_y.append(ry + y0)
      scat_c.append(np.tile(np.array(colors[c])[:3], (ry.size, 1)))
    y_edges.append(y_edges[-1] + h); y0 += h
  R = y_edges[-1]

  class_mean_s = _smooth(M.mean(0), SMOOTH_W)
  prof_smoothed = {}; sem_smoothed = {}; fracA_smoothed = {}

  # ---- determine per-cluster pileup y-axis bounds ----
  # Auto upper bound reproduces the original 1.15x behavior.
  auto_ymax = class_mean_s.max() * 1.15
  for c in ks:
    sel = np.where(labels == c)[0]
    prof = M[sel].mean(0)
    sem  = M[sel].std(0) / np.sqrt(max(1, sel.size))
    prof_smoothed[c] = _smooth(prof, SMOOTH_W)
    sem_smoothed[c]  = _smooth(sem,  SMOOTH_W)
    fracA_smoothed[c] = _smooth(V[sel].mean(0), SMOOTH_W)
    auto_ymax = max(auto_ymax, prof_smoothed[c].max() * 1.15)

  # Apply user overrides where provided; otherwise fall back to original values.
  ylo = 0.0 if pileup_ymin is None else float(pileup_ymin)
  yhi = float(auto_ymax) if pileup_ymax is None else float(pileup_ymax)
  if yhi <= ylo:
    # guard against degenerate/inverted bounds; revert upper to auto
    print(f"  WARN: pileup y-bounds invalid (ymin={ylo}, ymax={yhi}); "
          f"reverting ymax to auto={auto_ymax:.4g}", flush=True)
    yhi = float(auto_ymax)
    if yhi <= ylo:
      yhi = ylo + 1e-6

  n_k = len(ks)

  BASE_FS      = 13
  TITLE_FS     = 16
  SUPTITLE_FS  = 18
  AXLABEL_FS   = 14
  TICK_FS      = 12
  plt.rcParams.update({
    "font.size": BASE_FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": AXLABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
  })

  MIN_ROW_H = 1.9
  fig_h = max(10.0, MIN_ROW_H * n_k + 4.5)
  fig_w = 19.0

  fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

  gs = fig.add_gridspec(
    n_k, 4,
    width_ratios=[0.16, 3.0, 2.4, 1.7],
    wspace=0.45, hspace=0.55,
    left=0.055, right=0.965, top=0.93, bottom=0.10,
  )
  axSb  = fig.add_subplot(gs[:, 0])
  axL   = fig.add_subplot(gs[:, 1])
  axPer = [fig.add_subplot(gs[i, 2]) for i in range(n_k)]
  axPie = fig.add_subplot(gs[:, 3])

  sb_fs = max(9.0, min(13.0, 150.0 / max(1, n_k)))
  axSb.set_xlim(0, 1); axSb.set_ylim(R, 0)
  axSb.set_xticks([]); axSb.set_yticks([])
  axSb.set_ylabel("reads (grouped by cluster)", fontsize=AXLABEL_FS)

  axis_h_in = fig_h * (0.93 - 0.10)
  data_per_inch = R / max(axis_h_in, 1e-6)
  min_band_for_inside = (sb_fs / 72.0) * data_per_inch * 3.0

  outside_labels = []
  for c, c0, c1 in zip(ks, y_edges[:-1], y_edges[1:]):
    col = colors[c]
    axSb.axhspan(c0, c1, color=col)
    yc = (c0 + c1) / 2.0
    band_h = (c1 - c0)
    # Only show the cluster number to avoid overlapping text.
    txt = f"C{c}"
    if band_h >= (sb_fs / 72.0) * data_per_inch * 1.05:
      axSb.text(0.5, yc, txt,
                ha="center", va="center",
                fontsize=sb_fs,
                color="white" if _is_dark(col) else "black")
    else:
      outside_labels.append([yc, txt, col])

  if outside_labels:
    outside_labels.sort(key=lambda t: t[0])
    min_sep = (sb_fs / 72.0) * data_per_inch * 1.15
    for i in range(1, len(outside_labels)):
      if outside_labels[i][0] - outside_labels[i-1][0] < min_sep:
        outside_labels[i][0] = outside_labels[i-1][0] + min_sep
    for yc, txt, col in outside_labels:
      axSb.annotate(
        txt, xy=(0.0, yc), xytext=(-0.35, yc),
        textcoords="data", xycoords="data",
        ha="right", va="center", fontsize=max(8.0, sb_fs - 2),
        color=col if not _is_dark(col) else col,
        annotation_clip=False,
        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6,
                        shrinkA=0, shrinkB=0),
      )
  if scat_x:
    sx = np.concatenate(scat_x); sy = np.concatenate(scat_y); sc = np.vstack(scat_c)
    axL.scatter(sx, sy, c=sc, s=1.0, alpha=0.7, linewidths=0, marker=".")
  for e in y_edges[1:-1]:
    axL.axhline(e, color="0.5", lw=0.6, alpha=0.8)
  axL.axvline(0, color="k", ls=":", lw=0.6, alpha=0.5)
  axL.set_xlim(x[0], x[-1]); axL.set_ylim(R, 0); axL.set_yticks([])
  axL.set_xlabel("distance from motif center (bp)", fontsize=AXLABEL_FS)
  axL.tick_params(axis="x", labelsize=TICK_FS)
  axL.set_title("single reads by cluster (5'→ 3', region-oriented)", fontsize=TITLE_FS)

  per_fs = max(11.0, min(13.0, 130.0 / max(1, n_k)))
  ttl_fs = max(11.0, min(14.0, 140.0 / max(1, n_k)))
  for i, c in enumerate(ks):
    ax = axPer[i]
    sel = np.where(labels == c)[0]
    sel_size = sel.size
    ax.plot(x, class_mean_s, color="0.6", ls="--", lw=0.9, alpha=0.7)
    ax.plot(x, prof_smoothed[c], color=colors[c], lw=1.6)
    ax.fill_between(x, prof_smoothed[c] - sem_smoothed[c],
                    prof_smoothed[c] + sem_smoothed[c],
                    color=colors[c], alpha=0.25, linewidth=0)
    ax.axvline(0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(ylo, yhi)
    ax.set_title(f"C{c}  n={sel_size} ({100*sel_size/N:.1f}%)",
                 fontsize=ttl_fs, color=colors[c], pad=4)
    ax.tick_params(axis="both", labelsize=per_fs)
    if i < n_k - 1:
      ax.set_xticklabels([])
    else:
      ax.set_xlabel("bp from center", fontsize=per_fs + 1)
    ax.set_ylabel("m6A", fontsize=per_fs)

    axR = ax.twinx()
    axR.plot(x, fracA_smoothed[c], color="k", lw=1.0, alpha=0.65)
    axR.set_ylim(0, 1)
    axR.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    r_fs = max(9.0, min(per_fs, 11.0))
    axR.tick_params(axis="y", labelsize=r_fs, colors="0.25", pad=1, length=2)
    axR.set_ylabel("fraction A", fontsize=r_fs, color="0.25", labelpad=3)

  pie_sizes = [raw_counts[c] for c in ks]
  pie_colors = [colors[c] for c in ks]
  pie_pcts = [100 * s / N for s in pie_sizes]
  inline_th = 5.0 if n_k < 8 else 7.0
  pie_labels = [f"C{c}" if p >= inline_th else "" for c, p in zip(ks, pie_pcts)]
  pie_fs = max(10.0, min(13.0, 100.0 / max(1, n_k)))

  def _fmt(p): return (f"{p:1.1f}%") if p >= inline_th else ""

  axPie.set_anchor("N")
  wedges, _t, autotexts = axPie.pie(
    pie_sizes, labels=pie_labels, colors=pie_colors,
    autopct=_fmt, startangle=90, counterclock=False,
    pctdistance=0.72, labeldistance=1.12, radius=0.85,
    center=(0, 0.35),
    wedgeprops=dict(linewidth=0.6, edgecolor="white"),
    textprops=dict(fontsize=pie_fs),
  )
  for at, c in zip(autotexts, ks):
    at.set_fontsize(pie_fs)
    at.set_color("white" if _is_dark(colors[c]) else "black")

  leg_fs = max(9.0, min(12.0, 90.0 / max(1, n_k)))
  leg_labels = [f"C{c}  {p:.1f}%" for c, p in zip(ks, pie_pcts)]
  if n_k <= 6:
    ncol = 1
  elif n_k <= 12:
    ncol = 2
  else:
    ncol = 3
  leg = axPie.legend(
    wedges, leg_labels, loc="upper center",
    bbox_to_anchor=(0.5, -0.02), ncol=ncol, fontsize=leg_fs,
    frameon=False, handlelength=1.2, handletextpad=0.5,
    columnspacing=1.0, borderaxespad=0.0,
  )
  axPie.set_title(f"cluster share of {cls} (n={N})", fontsize=TITLE_FS)
  axPie.set_aspect("equal")
  axPie.set_xlim(-1.2, 1.2)
  axPie.set_ylim(-1.2, 1.4)
  axPie.axis("off")

  fig.suptitle(
    f"{tag} {cls}   KMeans k={k}   (fraction-A overlay, {SMOOTH_W}bp smoothing)",
    fontsize=SUPTITLE_FS, y=0.985,
  )

  png = os.path.join(outdir, f"k{k:02d}_paired.png")
  pdf = os.path.join(outdir, f"k{k:02d}_paired.pdf")
  fig.savefig(png, dpi=160, bbox_inches="tight")
  fig.savefig(pdf, bbox_inches="tight")
  plt.close(fig)
  return png


def cmd_paired_plots(cfg, tag, classes, kmin, kmax, overwrite_labels=False,
                     min_callable=MIN_CALLABLE_A,
                     min_methyl=MIN_METHYL_A,
                     min_callable_each_side=MIN_CALLABLE_EACH_SIDE,
                     scale_features=SCALE_FEATURES_DEFAULT,
                     pileup_ymin=PILEUP_YMIN_DEFAULT,
                     pileup_ymax=PILEUP_YMAX_DEFAULT):
  tp = tag_paths(cfg, tag)
  os.makedirs(tp["out_paired_root"], exist_ok=True)
  for cls in classes:
    bed = cfg["classes"][cls]
    outdir = os.path.join(tp["out_paired_root"], cls)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== paired-plots tag={tag} {cls} bed={os.path.basename(bed)} ===", flush=True)

    M, V = load_class_mod_val(tp["h5"], bed)

    M0 = M.shape[0]
    M, V, _keep = filter_reads_qc(M, V, min_callable, min_methyl, min_callable_each_side)
    print(f"  QC filter: kept {M.shape[0]}/{M0} reads ({100*M.shape[0]/max(1,M0):.1f}%)", flush=True)

    if M.shape[0] < kmax + 1:
      print(f"  SKIP: only {M.shape[0]} reads after QC", flush=True)
      continue

    X, _pca, n_pca_eff = featurize(M, V, N_PCA, RSTATE)
    feat_names = build_feature_names(n_pca_eff)

    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
      nbad = int((~ok).sum())
      print(f"  WARN: dropping {nbad} reads with non-finite features", flush=True)
      M = M[ok]; V = V[ok]; X = X[ok]

    print(f"  features {X.shape} (n_pca_eff={n_pca_eff}) scale={scale_features} "
          f"pileup_y=({pileup_ymin},{pileup_ymax})", flush=True)
    _ = feat_names

    for k in range(kmin, kmax + 1):
      labels, cached, _scaler = load_or_fit_labels_scaled(X, tp, cls, k, scale_features=scale_features)
      t0 = time.time()
      p = make_paired_figure_replot_style(M, V, labels, cls, k, outdir, tag,
                                          pileup_ymin=pileup_ymin,
                                          pileup_ymax=pileup_ymax)
      print(f"  k={k} labels_source={'cache' if cached else 'refit'} -> {p} ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------- feature-importance ----------------------------
def _sanitize(obj):
  if isinstance(obj, dict):
    return {int(k) if isinstance(k, (np.integer,)) else k: _sanitize(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize(v) for v in obj]
  if isinstance(obj, np.ndarray):
    return _sanitize(obj.tolist())
  if isinstance(obj, (np.integer,)):
    return int(obj)
  if isinstance(obj, (np.floating,)):
    return float(obj)
  if isinstance(obj, np.str_):
    return str(obj)
  return obj


def _fit_interp_with_labels(X, labels, method, feature_names):
  k = int(labels.max()) + 1
  centers = np.stack([X[labels == c].mean(axis=0) for c in range(k)], axis=0)

  km = KMeansInterp(
    ordered_feature_names=list(feature_names),
    feature_importance_method=method,
    n_clusters=k,
    init=centers,
    n_init=1,
    max_iter=1,
    random_state=RSTATE,
  )
  km.fit(X)
  km.labels_ = labels.astype(km.labels_.dtype)
  km.cluster_centers_ = centers.astype(km.cluster_centers_.dtype)

  if method == "wcss_min":
    km.feature_importances_ = km.get_feature_imp_wcss_min()
  elif method == "unsup2sup":
    km.feature_importances_ = km.get_feature_imp_unsup2sup(X)
  else:
    raise ValueError(method)
  return km


def _imp_matrix(imp_dict_method, feat_names, k):
  F = len(feat_names)
  M = np.zeros((k, F), dtype=float)
  name_to_col = {n: i for i, n in enumerate(feat_names)}
  for c, pairs in imp_dict_method.items():
    for feat, w in pairs:
      fn = str(feat)
      if fn in name_to_col:
        M[int(c), name_to_col[fn]] = float(w)
  return M


def _heatmap_grid(imp_dict, feat_names, k, cls, out_png, out_pdf):
  methods = list(imp_dict.keys())
  F = len(feat_names)
  n_meth = len(methods)

  col_w = 0.55
  row_h = 0.42
  fig_w = max(9.0, F * col_w + 3.5)
  fig_h = max(2.5, n_meth * (k * row_h + 1.4))
  fig, axes = plt.subplots(n_meth, 1, figsize=(fig_w, fig_h),
                           squeeze=False, constrained_layout=True)
  cmap = plt.get_cmap("Reds")

  for ax, m in zip(axes[:, 0], methods):
    M = _imp_matrix(imp_dict[m], feat_names, k)
    row_max = np.maximum(np.abs(M).max(axis=1, keepdims=True), 1e-12)
    Mn = np.abs(M) / row_max
    im = ax.imshow(Mn, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(F + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(F))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c}" for c in range(k)], fontsize=10)

    ax.set_title(
      f"{cls}   k={k}   method={m}   (color=|weight| row-normalized; number=raw weight)",
      fontsize=11, loc="left"
    )

    import matplotlib.patheffects as pe
    for c in range(k):
      for j in range(F):
        v = M[c, j]
        nv = Mn[c, j]
        av = abs(v)
        if av < 5e-3:
          s = "\u00b70"
          color = "0.55"
        elif av >= 1:
          s = f"{v:.2f}"
          color = "white" if nv > 0.55 else "black"
        elif av >= 0.01:
          s = f"{v:.3f}"
          color = "white" if nv > 0.55 else "black"
        else:
          s = f"{v:.1e}"
          color = "white" if nv > 0.55 else "black"
        halo = "black" if color == "white" else "white"
        ax.text(j, c, s, ha="center", va="center", fontsize=8, color=color,
                path_effects=[pe.withStroke(linewidth=1.4, foreground=halo)])

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("|weight| / row max", fontsize=9)
    cb.ax.tick_params(labelsize=8)

  fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
  fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
  plt.close(fig)


def _write_long_tsv(imp_dict, cls, k, out_tsv, top_n=None):
  with open(out_tsv, "w") as fh:
    fh.write("class\tk\tmethod\tcluster\trank\tfeature\tweight\n")
    for m, per_cluster in imp_dict.items():
      for c, pairs in per_cluster.items():
        for rank, (feat, w) in enumerate(pairs, start=1):
          if top_n is not None and rank > top_n:
            break
          fh.write(f"{cls}\t{k}\t{m}\t{c}\t{rank}\t{feat}\t{float(w):.6g}\n")


def cmd_feat_importance(cfg, tag, classes, kmin, kmax, skip_existing=False,
                        min_callable=MIN_CALLABLE_A,
                        min_methyl=MIN_METHYL_A,
                        min_callable_each_side=MIN_CALLABLE_EACH_SIDE,
                        scale_features=SCALE_FEATURES_DEFAULT):
  tp = tag_paths(cfg, tag)
  os.makedirs(tp["out_fi_root"], exist_ok=True)

  for cls in classes:
    bed = cfg["classes"][cls]
    outdir = os.path.join(tp["out_fi_root"], cls)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== feat-importance tag={tag} {cls} bed={os.path.basename(bed)} ===", flush=True)

    M, V = load_class_mod_val(tp["h5"], bed)

    M0 = M.shape[0]
    M, V, _keep = filter_reads_qc(M, V, min_callable, min_methyl, min_callable_each_side)
    print(f"  QC filter: kept {M.shape[0]}/{M0} reads ({100*M.shape[0]/max(1,M0):.1f}%)", flush=True)

    if M.shape[0] < kmax + 1:
      print(f"  SKIP: only {M.shape[0]} reads after QC", flush=True)
      continue

    X, _pca, n_pca_eff = featurize(M, V, N_PCA, RSTATE)
    feat_names = build_feature_names(n_pca_eff)

    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
      nbad = int((~ok).sum())
      print(f"  WARN: dropping {nbad} reads with non-finite features", flush=True)
      M = M[ok]; V = V[ok]; X = X[ok]

    if scale_features:
      scaler_local = StandardScaler().fit(X)
      Xs_local = scaler_local.transform(X)
    else:
      Xs_local = X

    print(f"  features {X.shape} (n_pca_eff={n_pca_eff}) scale={scale_features}", flush=True)

    for k in range(kmin, kmax + 1):
      json_path = os.path.join(outdir, f"k{k:02d}_feat_importance.json")
      png_path  = os.path.join(outdir, f"k{k:02d}_feat_importance.png")
      pdf_path  = os.path.join(outdir, f"k{k:02d}_feat_importance.pdf")
      tsv_path  = os.path.join(outdir, f"k{k:02d}_feat_importance_top.tsv")

      if skip_existing and os.path.exists(json_path) and os.path.exists(png_path):
        print(f"  skip existing k={k}", flush=True)
        continue

      t0 = time.time()
      labels, cached, scaler_cached = load_or_fit_labels_scaled(X, tp, cls, k, scale_features=scale_features)

      if scale_features and scaler_cached is not None:
        Xs = scaler_cached.transform(X)
      else:
        Xs = Xs_local

      counts = {int(c): int((labels == c).sum()) for c in range(int(labels.max()) + 1)}
      print(f"  k={k} labels_source={'cache' if cached else 'refit'} counts={counts}", flush=True)

      imp_by_method = {}
      for method_label, method_name, use_scaled in METHODS:
        Xu = Xs if use_scaled else X
        km = _fit_interp_with_labels(Xu, labels, method_name, feat_names)
        imp_by_method[method_label] = km.feature_importances_

      payload = dict(
        class_=cls,
        tag=tag,
        motif_slug=cfg["motif_slug"],
        motifs=list(MOTIFS),
        k=int(k),
        n_features=len(feat_names),
        feature_names=list(feat_names),
        cluster_sizes=counts,
        labels_source="cache" if cached else "refit",
        scale_features=bool(scale_features),
        importances=imp_by_method,
      )
      with open(json_path, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)

      _write_long_tsv(imp_by_method, cls, k, tsv_path)
      _heatmap_grid(imp_by_method, feat_names, k, cls, png_path, pdf_path)

      print(f"  k={k} done in {time.time()-t0:.1f}s -> {png_path}", flush=True)


def cmd_all(cfg, tag, classes, kmin, kmax, overwrite_labels=False, skip_existing_fi=False,
            min_callable=MIN_CALLABLE_A,
            min_methyl=MIN_METHYL_A,
            min_callable_each_side=MIN_CALLABLE_EACH_SIDE,
            scale_features=SCALE_FEATURES_DEFAULT,
            pileup_ymin=PILEUP_YMIN_DEFAULT,
            pileup_ymax=PILEUP_YMAX_DEFAULT):
  """Run BOTH paired-plots and feat-importance."""
  cmd_paired_plots(cfg, tag, classes, kmin, kmax,
                   overwrite_labels=overwrite_labels,
                   min_callable=min_callable,
                   min_methyl=min_methyl,
                   min_callable_each_side=min_callable_each_side,
                   scale_features=scale_features,
                   pileup_ymin=pileup_ymin,
                   pileup_ymax=pileup_ymax)
  cmd_feat_importance(cfg, tag, classes, kmin, kmax,
                      skip_existing=skip_existing_fi,
                      min_callable=min_callable,
                      min_methyl=min_methyl,
                      min_callable_each_side=min_callable_each_side,
                      scale_features=scale_features)


# ---------------------------- CLI helpers ----------------------------
def _resolve_tags(cfg, only_bams):
  """Which BAMs (tags) to process."""
  all_tags = list(cfg["bams"].keys())
  if not only_bams:
    return all_tags
  bad = [b for b in only_bams if b not in cfg["bams"]]
  if bad:
    raise KeyError(f"--only-bams unknown: {bad}. Known: {all_tags}")
  return list(only_bams)


def _resolve_classes(cfg, only_regions):
  """Which region sets (classes) to process."""
  all_cls = list(cfg["classes"].keys())
  if not only_regions:
    return all_cls
  bad = [c for c in only_regions if c not in cfg["classes"]]
  if bad:
    raise KeyError(f"--only-regions unknown: {bad}. Known: {all_cls}")
  return list(only_regions)


# ---------------------------- CLI ----------------------------
def main():
  ap = argparse.ArgumentParser(
    description="KMeans single-read pipeline: auto-runs every BAM x every region set."
  )
  ap.add_argument("--base", required=True,
                  help="Root output/data directory (required).")

  ap.add_argument("--min-callable", type=int, default=MIN_CALLABLE_A)
  ap.add_argument("--min-methyl", type=int, default=MIN_METHYL_A)
  ap.add_argument("--min-callable-each-side", type=int, default=MIN_CALLABLE_EACH_SIDE)

  ap.add_argument("--scale-features", action="store_true", default=SCALE_FEATURES_DEFAULT)
  ap.add_argument("--no-scale-features", action="store_false", dest="scale_features")

  # ---- per-cluster pileup y-axis bounds (default: auto, original behavior) ----
  ap.add_argument("--pileup-ymin", type=float, default=PILEUP_YMIN_DEFAULT,
                  help="lower bound for the per-cluster pileup (m6A) y-axis. "
                       "Default: None -> 0.0 (original behavior).")
  ap.add_argument("--pileup-ymax", type=float, default=PILEUP_YMAX_DEFAULT,
                  help="upper bound for the per-cluster pileup (m6A) y-axis. "
                       "Default: None -> auto 1.15x scaling (original behavior).")

  # ---- input registries (all required; no built-in defaults) ----
  ap.add_argument("--fasta", required=True,
                  help="Reference FASTA (required).")
  ap.add_argument("--bam", action="append", default=None, metavar="NAME=PATH",
                  help="register a BAM; repeatable; at least one required. "
                       "e.g. --bam sampleA=/x.bam")
  ap.add_argument("--regions", action="append", default=None, metavar="NAME=PATH",
                  help="register a region set; repeatable; at least one required. "
                       "e.g. --regions HH=/x.bed")
  ap.add_argument("--motifs", nargs="+", default=None, metavar="MOTIF",
                  help='motif(s), e.g. --motifs "A,0" (default: A,0). '
                       'Repeatable: --motifs "A,0" "CG,0". '
                       'Extracts/results are namespaced per motif.')

  # ---- subsetting the auto BAM x region grid ----
  ap.add_argument("--only-bams", nargs="+", default=None,
                  help="restrict to these BAM names (default: all registered)")
  ap.add_argument("--only-regions", nargs="+", default=None,
                  help="restrict to these region-set names (default: all registered)")

  # ---- h5 extraction is automatic by default; disable with --no-auto-build-h5 ----
  ap.add_argument("--auto-build-h5", action="store_true", default=True,
                  help="extract any missing/corrupt .h5 before analysis (default: on). "
                       "Extraction is cached, so existing valid h5 files are reused.")
  ap.add_argument("--no-auto-build-h5", action="store_false", dest="auto_build_h5",
                  help="do NOT extract; only run on pre-existing h5 files "
                       "(missing ones are skipped).")
  ap.add_argument("--extract-cores", type=int, default=EXTRACT_CORES,
                  help="cores for parse_bam.extract during (auto-)build")

  sub = ap.add_subparsers(dest="cmd", required=True)

  ap_bh = sub.add_parser("build-h5", help="only extract combined_basemods .h5 for BAM(s)")

  ap_pp = sub.add_parser("paired-plots", help="ONLY the per-cluster pileup figures")
  ap_pp.add_argument("--k-min", type=int, default=K_MIN)
  ap_pp.add_argument("--k-max", type=int, default=K_MAX)
  ap_pp.add_argument("--overwrite-labels", action="store_true")

  ap_fi = sub.add_parser("feat-importance", help="ONLY the feature-importance reports")
  ap_fi.add_argument("--k-min", type=int, default=K_MIN)
  ap_fi.add_argument("--k-max", type=int, default=K_MAX)
  ap_fi.add_argument("--skip-existing", action="store_true")

  ap_all = sub.add_parser("all", help="BOTH paired-plots and feat-importance")
  ap_all.add_argument("--k-min", type=int, default=K_MIN)
  ap_all.add_argument("--k-max", type=int, default=K_MAX)
  ap_all.add_argument("--overwrite-labels", action="store_true")
  ap_all.add_argument("--skip-existing-fi", action="store_true",
                      help="skip feat-importance outputs that already exist.")

  a = ap.parse_args()

  # override the global motif list BEFORE building cfg (slug depends on it)
  if a.motifs is not None:
    global MOTIFS
    MOTIFS = list(a.motifs)

  bams_override = _parse_name_path_pairs(a.bam)
  classes_override = _parse_name_path_pairs(a.regions)

  cfg = make_config(
    base=a.base,
    bams=bams_override,        # None -> DEFAULT_BAMS (empty -> error)
    fasta=a.fasta,             # required
    classes=classes_override,  # None -> DEFAULT_CLASSES (empty -> error)
  )

  tags = _resolve_tags(cfg, a.only_bams)
  classes = _resolve_classes(cfg, a.only_regions)

  print(f"BASE={cfg['base']}")
  print(f"FASTA={cfg['fasta']}")
  print(f"MOTIFS={MOTIFS}  (slug={cfg['motif_slug']})")
  print(f"BAMS (tags) = {list(cfg['bams'].keys())}")
  print(f"REGION SETS (classes) = {list(cfg['classes'].keys())}")
  print(f"PROCESSING BAMS  = {tags}")
  print(f"PROCESSING CLASSES = {classes}")
  print(f"SCALE_FEATURES={a.scale_features}")
  print(f"AUTO_BUILD_H5={a.auto_build_h5}")
  print(f"PILEUP_Y=({a.pileup_ymin},{a.pileup_ymax})", flush=True)

  # build-h5: extract every selected BAM, then stop.
  if a.cmd == "build-h5":
    cmd_build_h5(cfg, names=tags, cores=a.extract_cores)
    return

  # For analysis commands: loop over every selected BAM (tag) x classes.
  for tag in tags:
    print(f"\n############## BAM/tag = {tag}  (motif={cfg['motif_slug']}) ##############", flush=True)
    if a.auto_build_h5:
      ensure_h5(cfg, tag, cores=a.extract_cores)
    else:
      tp = tag_paths(cfg, tag)
      if not _h5_is_readable(tp["h5"]):
        print(f"[skip] no readable h5 for tag={tag} at {tp['h5']} "
              f"(remove --no-auto-build-h5 or run build-h5 first)", flush=True)
        continue

    if a.cmd == "paired-plots":
      cmd_paired_plots(cfg, tag, classes, a.k_min, a.k_max,
                       overwrite_labels=a.overwrite_labels,
                       min_callable=a.min_callable,
                       min_methyl=a.min_methyl,
                       min_callable_each_side=a.min_callable_each_side,
                       scale_features=a.scale_features,
                       pileup_ymin=a.pileup_ymin,
                       pileup_ymax=a.pileup_ymax)
    elif a.cmd == "feat-importance":
      cmd_feat_importance(cfg, tag, classes, a.k_min, a.k_max,
                          skip_existing=a.skip_existing,
                          min_callable=a.min_callable,
                          min_methyl=a.min_methyl,
                          min_callable_each_side=a.min_callable_each_side,
                          scale_features=a.scale_features)
    elif a.cmd == "all":
      cmd_all(cfg, tag, classes, a.k_min, a.k_max,
              overwrite_labels=a.overwrite_labels,
              skip_existing_fi=a.skip_existing_fi,
              min_callable=a.min_callable,
              min_methyl=a.min_methyl,
              min_callable_each_side=a.min_callable_each_side,
              scale_features=a.scale_features,
              pileup_ymin=a.pileup_ymin,
              pileup_ymax=a.pileup_ymax)


if __name__ == "__main__":
  main()
