#!/usr/bin/env bash
# Entry point for ScissilePIN prediction, generation, evaluation and fixed-recipe training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-python}"
GENERATOR_PYTHON="${GENERATOR_PYTHON:-$PYTORCH_PYTHON}"
METRICS_PYTHON="${METRICS_PYTHON:-$PYTORCH_PYTHON}"
DEVICE="${DEVICE:-cuda}"
FEATURE_PRECISION="${FEATURE_PRECISION:-bf16}"
SEEDS="${SEEDS:-0 1 2}"
MODE="${MODE:-selective}"
TARGET="${TARGET:-all}"
REQUEST="${REQUEST:-g1_original50}"
FROM_INITIALIZATION="${FROM_INITIALIZATION:-0}"
if [[ "$MODE" == unconditional ]]; then DEFAULT_TEMP=1.0; else DEFAULT_TEMP=1.2; fi
TEMPERATURE="${TEMPERATURE:-$DEFAULT_TEMP}"
SAMPLING_SEEDS="${SAMPLING_SEEDS:-2026111201 2026111202}"
PER_TEMPLATE="${PER_TEMPLATE:-400}"
RESUME="${RESUME:-0}"
PREDICTOR_LM_INIT="${PREDICTOR_LM_INIT:-$ROOT/predictor/initialization/lmw_256_8_iso.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs}"
for seed in $SEEDS; do
  case "$seed" in 0|1|2) ;; *) echo 'SEEDS must contain only 0, 1 and 2.' >&2; exit 2;; esac
done
help() {
  cat <<'EOF'
Usage: bash run.sh COMMAND

  verify           Verify files, labelled datasets and disjoint sequence splits.
  smoke            Load all predictor weights and sample 50 peptides per generator.
  predictor        Extract ESM features; predict val/test/OOD; evaluate all three seeds.
  generator        Evaluate development NLL and generate ScissilePIN generator standard-DPO pools.
  score-generated  Score generated pools using the three-seed ScissilePIN predictor ensemble.
  all              Run verify, predictor, generator and score-generated.
  train-predictor  Train the ScissilePIN predictor.
  train-generator  Train the complete ScissilePIN generator pipeline.
  train-generator-dpo  Train DPO from the included supervised weights.

Environment variables:
  PYTORCH_PYTHON, GENERATOR_PYTHON, METRICS_PYTHON   Python executables
  DEVICE=cuda, FEATURE_PRECISION=bf16              Device and predictor/ESM precision
  SEEDS="0 1 2", TEMPERATURE=1.2, MODE=selective     Experiment settings
  TARGET=all, REQUEST=g1_original50              Select one MMP or all 18
  FROM_INITIALIZATION=0                          Start at packaged mixture when set to 1
  PER_TEMPLATE=400, OUTPUT_DIR=./outputs           Sampling count and output location
  SAMPLING_SEEDS="2026111201 2026111202"            Separate sampling replicates
  FEATURE_DIR=./outputs/features                   Reuse extracted ESM features
  POOL_DIR=./outputs/generator/pools               Input pools for score-generated
  RESUME=0                                       Set 1 to resume training
  PREDICTOR_LM_INIT=./predictor/initialization/lmw_256_8_iso.pt
                                                 Task-LM initialization

CPU feature extraction: DEVICE=cpu FEATURE_PRECISION=fp32 bash run.sh predictor
EOF
}
if [[ "${1:-help}" == help || "${1:-}" == --help || "${1:-}" == -h ]]; then help; exit 0; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
FEATURE_DIR="${FEATURE_DIR:-$OUT/features}"
verify() { "$METRICS_PYTHON" evaluation/verify_data.py; }
smoke() {
  "$PYTORCH_PYTHON" evaluation/smoke_predictor.py
  "$GENERATOR_PYTHON" evaluation/smoke_generator.py
  mkdir -p "$OUT/smoke/pools"
  local target=MMP1
  for seed in $SEEDS; do
    "$GENERATOR_PYTHON" generator/sample.py --target "$target" --request "$REQUEST" --checkpoint "generator/checkpoints/generator_SEED${seed}.pt" --device "$DEVICE" --seed 2026111201 --per-template 1 --output "$OUT/smoke/pools/PRISM_generator_seed${seed}_sample2026111201_T1.2_selective.csv"
  done
  "$METRICS_PYTHON" evaluation/generator_quality.py --pools "$OUT/smoke/pools" --output "$OUT/smoke/quality"
}
predictor() {
  mkdir -p "$FEATURE_DIR" "$OUT/predictor/predictions"
  for split in val test ood; do
    input="data/benchmark/$split.csv"
    [[ "$split" != ood ]] || input=data/ood/ood.csv
    if [[ ! -f "$FEATURE_DIR/$split.npz" ]]; then
      "$PYTORCH_PYTHON" predictor/extract_features.py --input "$input" --output "$FEATURE_DIR/$split.npz" --device "$DEVICE" --precision "$FEATURE_PRECISION"
    fi
    for seed in $SEEDS; do
      "$PYTORCH_PYTHON" predictor/predict.py --checkpoint "predictor/checkpoints/seed${seed}.pt" --input "$FEATURE_DIR/$split.npz" --output "$OUT/predictor/predictions/${split}_seed${seed}.npz" --device "$DEVICE"
    done
  done
  "$METRICS_PYTHON" evaluation/evaluate_selection.py --predictions "$OUT/predictor/predictions" --output "$OUT/predictor/metrics" --seeds $SEEDS
}
generator() {
  mkdir -p "$OUT/generator/nll" "$OUT/generator/pools"
  local targets
  if [[ "$TARGET" == all ]]; then targets="$("$GENERATOR_PYTHON" -c 'import json; print(" ".join(json.load(open("generator/data/target_order.json"))))')"; else targets="$TARGET"; fi
  for seed in $SEEDS; do
    "$GENERATOR_PYTHON" generator/evaluate_checkpoint.py --target "$TARGET" --checkpoint "generator/checkpoints/generator_SEED${seed}.pt" --device "$DEVICE" --output "$OUT/generator/nll/nll_seed${seed}.json"
    for target in $targets; do
    for rep in $SAMPLING_SEEDS; do
      "$GENERATOR_PYTHON" generator/sample.py --target "$target" --request "$REQUEST" --checkpoint "generator/checkpoints/generator_SEED${seed}.pt" --device "$DEVICE" --seed "$rep" --temperature "$TEMPERATURE" --mode "$MODE" --per-template "$PER_TEMPLATE" --output "$OUT/generator/pools/PRISM_generator_${target}_${REQUEST}_seed${seed}_sample${rep}_T${TEMPERATURE}_${MODE}.csv"
    done
    done
  done
  "$METRICS_PYTHON" evaluation/summarize_nll.py --input "$OUT/generator/nll"
  "$METRICS_PYTHON" evaluation/generator_quality.py --pools "$OUT/generator/pools" --output "$OUT/generator/quality"
}
score_generated() {
  "$PYTORCH_PYTHON" evaluation/score_generated.py --pools "${POOL_DIR:-$OUT/generator/pools}" --output "$OUT/generator/prism_scores" --device "$DEVICE" --precision "$FEATURE_PRECISION"
}
train_predictor() {
  "$PYTORCH_PYTHON" predictor/train.py --lm-init "$PREDICTOR_LM_INIT" --check-inputs
  mkdir -p "$FEATURE_DIR"
  for split in train val; do
    if [[ ! -f "$FEATURE_DIR/$split.npz" ]]; then
      "$PYTORCH_PYTHON" predictor/extract_features.py --input "data/benchmark/$split.csv" --output "$FEATURE_DIR/$split.npz" --device "$DEVICE" --precision "$FEATURE_PRECISION"
    fi
  done
  local extra=()
  [[ "$RESUME" != 1 ]] || extra+=(--resume)
  for seed in $SEEDS; do
    "$PYTORCH_PYTHON" predictor/train.py --seed "$seed" --device "$DEVICE" --features "$FEATURE_DIR" --lm-init "$PREDICTOR_LM_INIT" --output "$OUT/predictor/training/seed${seed}" ${extra[@]+"${extra[@]}"}
  done
}
train_generator() {
  local extra=()
  [[ "$FROM_INITIALIZATION" != 1 ]] || extra+=(--from-initialization)
  [[ "$RESUME" != 1 ]] || extra+=(--resume)
  for seed in $SEEDS; do
    "$GENERATOR_PYTHON" generator/train_pipeline.py --seed "$seed" --device "$DEVICE" --output "$OUT/generator/training/seed${seed}" ${extra[@]+"${extra[@]}"}
  done
}
train_generator_dpo() {
  local extra=()
  [[ "$RESUME" != 1 ]] || extra+=(--resume)
  for seed in $SEEDS; do
    local workers=1
    [[ "$seed" != 0 || "$DEVICE" != cuda ]] || workers=2
    "$GENERATOR_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$workers" generator/train.py --seed "$seed" --device "$DEVICE" --output "$OUT/generator/dpo/seed${seed}" ${extra[@]+"${extra[@]}"}
  done
}
case "$1" in
  verify) verify;; smoke) smoke;; predictor) predictor;; generator) generator;;
  score-generated) score_generated;; all) verify; predictor; generator; score_generated;;
  train-predictor) train_predictor;; train-generator) train_generator;;
  train-generator-dpo) train_generator_dpo;; *) help; exit 2;;
esac
