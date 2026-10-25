#!/bin/bash
# =====================================================================================
#  config.sh  -  cluster settings. Fill in the two CHANGE_ME values, save, done.
#  (WHAT to run is in experiments.yaml; this file is only about the cluster.)
#
#  How to find the two values (run these on the cluster after `ssh`):
#     sacctmgr show assoc user=$USER format=Account,Partition,QOS -P   -> SLURM_ACCOUNT
#     sinfo -o "%P %G %l"                                             -> a partition with GPUs
# =====================================================================================

# ---- SLURM (required) ---------------------------------------------------------------
SLURM_ACCOUNT="CHANGE_ME"        # your account / project name
SLURM_PARTITION="CHANGE_ME"      # a GPU partition (column %G of sinfo shows "gpu:...")
SLURM_QOS=""                     # leave empty unless the cluster requires a QOS

# ---- Where things go ------------------------------------------------------------------
DATA_DIR="$HOME/tda_data"        # datasets (~25 GB without SUN397). Use a scratch/work
                                 # folder here if your home quota is small.
ENV_NAME="tda"                   # name of the conda environment created by setup

# ---- Resources per job ---------------------------------------------------------------
CPUS=8                           # 8 = the number of data-loader workers in the original code
MEM="32G"
TIME_CHECK="01:00:00"            # the quick check job
TIME_DEFAULT="12:00:00"          # experiments without a `time:` in experiments.yaml
BACKBONES="RN50 ViT-B/16"        # backbones tested by the equivalence check
