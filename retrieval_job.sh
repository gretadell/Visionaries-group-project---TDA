#!/bin/bash
# retrieval_job.sh - retrieval analysis (which cache entries drive TDA's predictions) on the Bocconi cluster.
#
#   1. put retrieval_analysis.py and this file in ~/TDA_replication (next to streams.py, context_cache.py)
#   2. sed -i 's/\r$//' retrieval_job.sh retrieval_analysis.py      (Windows line endings)
#   3. set your student ID below, then:  mkdir -p logs && sbatch retrieval_job.sh
#   4. follow it with:  tail -f logs/retrieval_<jobid>.out
#
# Needs results/features/core50_350__RN50.pt and core50_350__ViT-B-16.pt (already there from the 350 runs).
# CORe50 128 (core50__*.pt) is run too if those files exist: it is the "little background" comparison.
# About 3 min per run on an A100: 350 = 2 backbones x 2 modes x 3 orders x 3 seeds = 36 runs (~2 h).
# Finished runs are skipped, so the job can simply be resubmitted after a timeout.

#SBATCH --job-name=tda_retrieval
#SBATCH --account=CHANGE_ME_STUDENT_ID
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/retrieval_%j.out

set -e
cd ~/TDA_replication
mkdir -p logs
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tda

for DS in core50_350 core50; do
  for BB in RN50 ViT-B/16; do
    if [ ! -f "results/features/${DS}__${BB/\//-}.pt" ]; then
      echo "skip $DS $BB: feature file not found"; continue
    fi
    python retrieval_analysis.py run --dataset "$DS" --backbone "$BB" --cfg-like caltech101 \
           --sessions 3,7,10 --order s0,s2,s3 --mode tda,ctxx --repeats 3
  done
done

python retrieval_analysis.py summary
echo "done: send results/retrieval/retrieval_summary.md and retrieval_summary.csv"
