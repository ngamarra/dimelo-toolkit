#!/bin/bash
#SBATCH --job-name=single_read_kmeans
#SBATCH --partition=owners,normal,altemose
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00

export GROUP_HOME=/home/groups/altemose
module load devel python/3.12.1
source "$GROUP_HOME/envs/dimelo-toolkit/bin/activate"
export PATH=/home/groups/altemose/envs/dimelo-toolkit/bin:$PATH
export PYTHONPATH=/home/groups/altemose/kmeans_interp_src/kmeans-feature-importance:${PYTHONPATH:-}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ../single_read_kmeans.py \
  --base . \
  --fasta /oak/stanford/groups/altemose/references/fastas/chm13v2.0.fasta \
  --bam CTCF_targetting=/oak/stanford/groups/altemose/data/20250110_NG_one_pot/barcode17.merged.sorted.bam \
  --bam CTCF_targetting_mitosis=/oak/stanford/groups/altemose/data/20250130_NG_onepot/merged.sorted.bam \
  --regions HH=/scratch/groups/altemose/grips/czhao/motif_checked/intersections_150bp/CTCFhigh_RAD21high.w150.bed \
  --regions HL=/scratch/groups/altemose/grips/czhao/motif_checked/intersections_150bp/CTCFhigh_RAD21low.w150.bed \
  --regions LH=/scratch/groups/altemose/grips/czhao/motif_checked/intersections_150bp/CTCFlow_RAD21high.w150.bed \
  --regions LL=/scratch/groups/altemose/grips/czhao/motif_checked/intersections_150bp/CTCFlow_RAD21low.w150.bed \
  --pileup-ymax 0.2 \
  --extract-cores $SLURM_CPUS_PER_TASK \
  all \
  --k-min 4 --k-max 10 
