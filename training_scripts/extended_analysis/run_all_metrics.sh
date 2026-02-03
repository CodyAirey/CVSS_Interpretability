#!/bin/bash
#SBATCH --job-name=distilbert-metrics-arr
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=50GB
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=aoraki_gpu
#SBATCH --array=0-7
#SBATCH --output=rubbishlogs/slurm-%x-%A-%a.log   # fallback / workaround; we also write our own descriptive log below

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cvss_predictions
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# Fixed parameters
MODEL="distilbert"
SEED=123
DATA_FILE="small_cve_ids_3.1_header.csv"
TEST_SIZE=0.20
KFOLD=5

# Suffix ONLY for display/logging
DISPLAY_MODEL="$MODEL"
if [ "$KFOLD" -gt 0 ]; then
  DISPLAY_MODEL="${MODEL}-cv"
fi


# Metric order you care about
# metrics=(av ac pr ui s c i a)
# metrics=(c a)
metrics=(pr)


# Resolve this task's metric
idx=${SLURM_ARRAY_TASK_ID}
m=${metrics[$idx]}

# Zero-pad the index so filename order matches metric order (00, 01, 02, …)
idxp=$(printf "%02d" "$idx")

# Group by model+seed, then order by your metric array index
LOGFILE="logs/5k-CV-full-${DISPLAY_MODEL}-${SEED}-${idxp}-${m}-test${TEST_SIZE}.log"
exec >"$LOGFILE" 2>&1

echo ">>> Running metric=$m (index $idx) seed=$SEED model(display)=$DISPLAY_MODEL model(arg)=$MODEL data_file=$DATA_FILE test_size=$TEST_SIZE kfold=$KFOLD"
echo ">>> logfile: $LOGFILE"

python train.py \
  --data_file "$DATA_FILE" \
  --metric "$m" \
  --model "$MODEL" \
  --epochs 3 \
  --train_batch 8 \
  --eval_batch 4 \
  --lr 3e-5 \
  --val_size 0 \
  --test_size "$TEST_SIZE" \
  --seed "$SEED" \
  # --kfold "$KFOLD" \
