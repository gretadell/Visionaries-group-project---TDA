#!/bin/bash
# =====================================================================================
#  run.sh  -  single entry point on the cluster.      Usage:  bash run.sh <command>
#  Cluster settings: config.sh · what to run: experiments.yaml · method: tda_replica.py
# =====================================================================================
set -eo pipefail

# Inside a SLURM job this script runs from a spool copy, so the repo path is passed in TDA_HOME
if [[ -n "${SLURM_JOB_ID:-}" && -n "${TDA_HOME:-}" ]]; then HERE="$TDA_HOME"
else HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
source "$HERE/config.sh"

export TDA_HOME="$HERE"
export REPO_DIR="$HERE/TDA"               # official repository (cloned by `setup`)
export RESULTS_DIR="$HERE/results"        # everything you want to look at ends up here
export LOG_DIR="$HERE/logs"               # full logs (progress bars, tracebacks)
export DATA_DIR BACKBONES
PAPER_EXPERIMENTS="paper_cd paper_ood ablation_components ablation_shots"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

usage() {
cat <<'EOF'

  TDA replication on the cluster  -  bash run.sh <command>

  ONE-TIME PREPARATION (login node)
    setup             conda env with the authors' versions, official repo, CLIP weights
    data [names]      download the datasets in the background (safe to close the terminal)
    data-log          follow the download log (Ctrl+C stops watching, not the download)

  EXPERIMENTS (each one is submitted to SLURM and runs on a GPU node)
    check             quick job: test-set sizes + equivalence test replica vs official code
    all               check, then the paper reproduction: paper_cd paper_ood ablation_components ablation_shots
    run NAME [NAME..] run experiments of experiments.yaml (one job per backbone)
    list              show the experiments defined in experiments.yaml

  MONITORING AND RESULTS
    status            your jobs in the queue + the last progress line of each log
    report            rebuild results/REPORT.md
    cancel            cancel all your TDA jobs

EOF
}

activate_env() {
    module load miniconda3 >/dev/null 2>&1 || true     # cluster module, if it exists
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found. Install Miniconda in your home (README, Troubleshooting), then: bash run.sh setup"
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

require_ready() {
    if [[ "$SLURM_ACCOUNT" == "CHANGE_ME" || "$SLURM_PARTITION" == "CHANGE_ME" ]]; then
        echo "ERROR: open config.sh and set SLURM_ACCOUNT and SLURM_PARTITION first."; exit 1
    fi
    if [[ ! -d "$REPO_DIR" ]]; then echo "ERROR: run 'bash run.sh setup' first."; exit 1; fi
    activate_env "$ENV_NAME"
}

# submit <job name> <time limit> <dependency or ""> <arguments for experiments.py...>  -> prints the job id
submit() {
    local name="$1" time="$2" dep="$3"; shift 3
    local opts=(--parsable --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION"
                --nodes 1 --ntasks 1 --gpus=1 --cpus-per-task "$CPUS" --mem "$MEM" --time "$time"
                --job-name "$name" --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.out"
                --export "ALL,TDA_HOME=$HERE")
    [[ -n "$SLURM_QOS" ]] && opts+=(--qos "$SLURM_QOS")
    [[ -n "$dep" ]] && opts+=(--dependency "$dep")
    sbatch "${opts[@]}" "$HERE/run.sh" _job "$@"
}

# submit_experiment <name> <dependency or "">  -> one job per backbone, prints "id(backbone) ..."
submit_experiment() {
    local name="$1" dep="$2" out=() bb time id
    while read -r bb time; do
        id=$(submit "tda-$name-${bb//\//-}" "$time" "$dep" run "$name" --backbone "$bb")
        out+=("$id($bb)")
    done < <(python "$HERE/experiments.py" plan "$name" --default-time "$TIME_DEFAULT")
    echo "${out[@]}"
}

cmd="${1:-help}"
case "$cmd" in
    setup)
        step() { echo; echo "---------- $* ----------"; }
        step "1/4 conda environment '$ENV_NAME' (Python 3.7, as in the paper)"
        module load miniconda3 >/dev/null 2>&1 || true
        command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found (README, Troubleshooting)."; exit 1; }
        eval "$(conda shell.bash hook)"
        if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then echo "already exists"
        else conda create -y -n "$ENV_NAME" python=3.7; fi
        conda activate "$ENV_NAME"

        step "2/4 Python packages (torch 1.12.1 + CUDA 11.3, the authors' versions)"
        if python -c "import torch; assert torch.__version__.startswith('1.12.1')" 2>/dev/null; then
            echo "PyTorch 1.12.1 already installed"
        else   # pip wheels ship their own CUDA runtime: no cluster CUDA module needed
            pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
                --extra-index-url https://download.pytorch.org/whl/cu113
        fi
        pip install -q ftfy regex tqdm pyyaml scipy "gdown==4.7.1" "wandb==0.15.12" pandas matplotlib

        step "3/4 official repository -> TDA/"
        [[ -d "$REPO_DIR/.git" ]] || git clone https://github.com/kdiAAA/TDA.git "$REPO_DIR"
        git -C "$REPO_DIR" log -1 --format="official code commit: %H (%cd)"

        step "4/4 CLIP weights (downloaded now, so GPU jobs need no internet)"
        (cd "$REPO_DIR" && python -c "import clip; [clip.load(m, device='cpu') for m in ['RN50', 'ViT-B/16']]; print('ok')")
        echo; echo "Setup complete. Next step:  bash run.sh data" ;;

    data)
        shift || true
        activate_env "$ENV_NAME"
        nohup python "$HERE/prepare_data.py" "$@" > "$LOG_DIR/prepare_data.log" 2>&1 &
        echo "Download started in the background (PID $!). You can close the terminal."
        echo "Follow it with:  bash run.sh data-log" ;;

    data-log)
        tail -n 30 -f "$LOG_DIR/prepare_data.log" ;;

    check)
        require_ready
        echo "Submitted check job $(submit tda-check "$TIME_CHECK" "" check)."
        echo "When it finishes, read results/REPORT.md (follow it with: bash run.sh status)." ;;

    run)
        shift || true
        [[ $# -gt 0 ]] || { echo "Usage: bash run.sh run NAME [NAME...]   (names: bash run.sh list)"; exit 1; }
        require_ready
        for name in "$@"; do python "$HERE/experiments.py" plan "$name" > /dev/null; done   # stops on a typo
        for name in "$@"; do echo "  $name: $(submit_experiment "$name" "")"; done
        echo "Follow with:  bash run.sh status      Results:  results/REPORT.md" ;;

    all)
        require_ready
        check=$(submit tda-check "$TIME_CHECK" "" check)
        echo "Submitted:"; echo "  check: $check"
        for name in $PAPER_EXPERIMENTS; do
            echo "  $name: $(submit_experiment "$name" "afterok:$check")   (starts after the check succeeds)"
        done
        echo "Follow with:  bash run.sh status      Results:  results/REPORT.md" ;;

    list)
        activate_env "$ENV_NAME"
        python "$HERE/experiments.py" list ;;

    status)
        echo "== Your jobs =="
        squeue -u "$USER" -o "%.10i %.32j %.9T %.10M %.10l %R" || true
        echo; echo "== Last progress line of the most recent logs =="
        for f in $(ls -t "$LOG_DIR"/*.log "$LOG_DIR"/*.out 2>/dev/null | head -n 8); do
            printf "%-45s %s\n" "$(basename "$f")" "$(tr '\r' '\n' < "$f" | grep -v '^\s*$' | tail -n 1 | cut -c1-100)"
        done
        echo; echo "Report: $RESULTS_DIR/REPORT.md  (rebuilt at the end of every job, or with: bash run.sh report)" ;;

    report)
        activate_env "$ENV_NAME"
        python "$HERE/experiments.py" report ;;

    cancel)
        ids=$(squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /^tda-/ {print $1}')
        if [[ -n "$ids" ]]; then scancel $ids && echo "Cancelled: $ids"; else echo "No TDA jobs in the queue."; fi ;;

    _job)   # internal: what every SLURM job executes on the GPU node
        shift
        activate_env "$ENV_NAME"
        export WANDB_MODE=disabled PYTHONUNBUFFERED=1
        cd "$HERE"
        echo "=============================================================="
        echo " Job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) - $(date)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
        echo " Step: experiments.py $*"
        echo "=============================================================="
        status=0
        python experiments.py "$@" || status=$?
        python experiments.py report --quiet || echo "(report not updated)"   # keep REPORT.md up to date
        echo "Done - $(date) - exit code $status"
        exit $status ;;

    help|-h|--help|*)
        usage ;;
esac
