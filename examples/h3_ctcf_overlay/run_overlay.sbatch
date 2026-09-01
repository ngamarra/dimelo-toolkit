#!/bin/bash
#SBATCH -p owners,normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -J h3_ctcf_overlay
#SBATCH -o /scratch/users/ngamarra/ngamarra_dimelo_fork/examples/h3_ctcf_overlay/overlay.%j.out
#SBATCH -e /scratch/users/ngamarra/ngamarra_dimelo_fork/examples/h3_ctcf_overlay/overlay.%j.out
set -uo pipefail

FORK=/scratch/users/ngamarra/ngamarra_dimelo_fork
MM=/scratch/users/ngamarra/envs/micromamba
ENV=/home/groups/altemose/envs/ngamarra_dimelo_fork
export MAMBA_ROOT_PREFIX=/scratch/users/ngamarra/mamba_root
export CONDA_PKGS_DIRS=/scratch/users/ngamarra/mamba_root/pkgs
RUN="$MM run -p $ENV"

cd "$FORK"
echo "### unit tests for dimelo.track_overlay"
$RUN python -m dimelo.test.test_track_overlay || { echo "UNIT TESTS FAILED"; exit 1; }

echo "### demo: overlay H3 tracks on targeting CTCF sites"
$RUN python examples/h3_ctcf_overlay/run_overlay.py "$@"

echo "### DONE"
ls -la examples/h3_ctcf_overlay/out/ 2>/dev/null
