#!/bin/bash
# Make src/ a top-level import root so bare imports like `offline_evals` and
# `scripts.eval.tasks` resolve. pip install -e . only registers olmo_core*.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
#
# HuggingFace-Native Finetuning Pipeline with Continuous-Sparsification (CS)
# Expert Selection (Savarese, Silva & Maire, "Winning the Lottery with
# Continuous Sparsification," NeurIPS 2020), mapped onto EMO routed experts.
#
# Differences from hf_finetune_with_selective_easy_ep.sh:
#   * Selection is a short *optimization* loop (forward+backward over calibration
#     data) that trains a per-expert mask logit with an L1 penalty while annealing
#     a sigmoid temperature beta upward — not a one-shot scoring pass.
#   * The learned boolean decision is baked into the same structural prune kernel,
#     so the emitted artifact is identical in form to the other methods and the
#     finetune + eval tail below is unchanged.
#   * Calibration uses the *train* split (CS needs masked labels for a task loss).
#
# Usage:
#   ./scripts/selective_hf/hf_finetune_with_selective_continuous.sh \
#       --model /path/to/model \
#       --task arc_challenge \
#       --selective-keep-k 32 \
#       --num-shared-experts 1 \
#       --num-selective-examples 256 \
#       --selection-mode per_layer \
#       --base-dir /path/to/output \
#       --relative-dir <run_subdir> \
#       --num-gpus 4 \
#       --run-name <job_name>
#

set -e

# Default values
MODEL=""
TASK=""
SELECTIVE_KEEP_K=""
NUM_SHARED_EXPERTS=0
RELATIVE_DIR=""
BASE_DIR=""
NUM_GPUS=1
NUM_EPOCHS=3
NUM_CHECKPOINTS=5
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
SKIP_SELECTIVE=false
SELECTED_MODEL=""
LEARNING_RATE=5e-5
RUN_NAME=""
NUM_SELECTIVE_EXAMPLES=""
NUM_SELECTIVE_SEED=""
NUM_SHOTS_SELECTIVE=""
NUM_SHOTS_EVAL=""
TRUST_REMOTE_CODE=false
# CS-specific selection hyperparameters.
SELECTION_MODE="per_layer"
CS_LAMBDA=""
CS_LR=""
CS_NUM_STEPS=""
CS_BETA_START=""
CS_BETA_END=""
CS_S_INIT=""
CS_SELECT_BATCH_SIZE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"; shift 2 ;;
        --task)
            TASK="$2"; shift 2 ;;
        --selective-keep-k)
            SELECTIVE_KEEP_K="$2"; shift 2 ;;
        --num-shared-experts)
            NUM_SHARED_EXPERTS="$2"; shift 2 ;;
        --relative-dir)
            RELATIVE_DIR="$2"; shift 2 ;;
        --base-dir)
            BASE_DIR="$2"; shift 2 ;;
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --num-epochs)
            NUM_EPOCHS="$2"; shift 2 ;;
        --num-checkpoints)
            NUM_CHECKPOINTS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --skip-selective)
            SKIP_SELECTIVE=true; shift ;;
        --selected-model)
            SELECTED_MODEL="$2"; shift 2 ;;
        --learning-rate)
            LEARNING_RATE="$2"; shift 2 ;;
        --run-name)
            RUN_NAME="$2"; shift 2 ;;
        --num-selective-examples)
            NUM_SELECTIVE_EXAMPLES="$2"; shift 2 ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE=true; shift ;;
        --num-selective-seed)
            NUM_SELECTIVE_SEED="$2"; shift 2 ;;
        --num-shots-selective)
            NUM_SHOTS_SELECTIVE="$2"; shift 2 ;;
        --num-shots-eval)
            NUM_SHOTS_EVAL="$2"; shift 2 ;;
        # CS-specific.
        --selection-mode)
            SELECTION_MODE="$2"; shift 2 ;;
        --cs-lambda)
            CS_LAMBDA="$2"; shift 2 ;;
        --cs-lr)
            CS_LR="$2"; shift 2 ;;
        --cs-num-steps)
            CS_NUM_STEPS="$2"; shift 2 ;;
        --cs-beta-start)
            CS_BETA_START="$2"; shift 2 ;;
        --cs-beta-end)
            CS_BETA_END="$2"; shift 2 ;;
        --cs-s-init)
            CS_S_INIT="$2"; shift 2 ;;
        --cs-select-batch-size)
            CS_SELECT_BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model                    Path to the full (unselected) HF model"
            echo "  --task                     Task name (arc_challenge, mmlu, etc.)"
            echo "  --selective-keep-k         Total experts to keep per layer (per_layer/global)"
            echo "  --relative-dir             Relative output subdirectory"
            echo "  --base-dir                 Base output directory"
            echo "  --run-name                 Run name for logging"
            echo ""
            echo "Optional:"
            echo "  --num-shared-experts       Shared experts to keep (default: 0)"
            echo "  --num-gpus                 GPUs for finetuning (default: 1)"
            echo "  --num-epochs               Finetuning epochs (default: 3)"
            echo "  --num-checkpoints          Checkpoints to save (default: 5)"
            echo "  --batch-size               Global batch size (default: 4)"
            echo "  --micro-batch-size         Per-device batch size (default: 1)"
            echo "  --skip-selective           Skip selection; requires --selected-model"
            echo "  --selected-model           Path to an already-selected model"
            echo "  --learning-rate            Learning rate (default: 5e-5)"
            echo ""
            echo "CS selection (all optional; script defaults apply if omitted):"
            echo "  --selection-mode           per_layer | global | threshold (default: per_layer)"
            echo "  --cs-lambda                L1 penalty weight on soft gates"
            echo "  --cs-lr                    Adam LR for mask logits"
            echo "  --cs-num-steps             Calibration optimization steps"
            echo "  --cs-beta-start            Initial sigmoid temperature"
            echo "  --cs-beta-end              Final (annealed) sigmoid temperature"
            echo "  --cs-s-init                Initial mask logit value"
            echo "  --cs-select-batch-size     Batch size for the calibration loop"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ] && [ "$SKIP_SELECTIVE" = false ]; then
    echo "Error: --model is required unless --skip-selective is set"; exit 1
fi
if [ "$SKIP_SELECTIVE" = true ] && [ -z "$SELECTED_MODEL" ]; then
    echo "Error: --selected-model is required when --skip-selective is set"; exit 1
fi
if [ "$SKIP_SELECTIVE" = false ] && [ -z "$SELECTIVE_KEEP_K" ]; then
    echo "Error: --selective-keep-k is required unless --skip-selective is set"; exit 1
fi
if [ -z "$TASK" ]; then
    echo "Error: --task is required"; exit 1
fi
if [ -z "$RELATIVE_DIR" ]; then
    echo "Error: --relative-dir is required"; exit 1
fi
if [ -z "$BASE_DIR" ]; then
    echo "Error: --base-dir is required"; exit 1
fi
if [[ "$RELATIVE_DIR" != *"$TASK"* ]]; then
    echo "ERROR: --relative-dir does not contain the task name '$TASK'"; exit 1
fi
if [ -z "$RUN_NAME" ]; then
    echo "Error: --run-name is required"; exit 1
fi
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "Error: --batch-size must be a multiple of --num-gpus"; exit 1
fi
if (( BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "Error: --batch-size must be a multiple of --micro-batch-size"; exit 1
fi

OUTPUT_DIR="${BASE_DIR}/${RELATIVE_DIR}"
mkdir -p "$OUTPUT_DIR"

if [ -z "$SELECTED_MODEL" ]; then
    SELECTED_MODEL="${OUTPUT_DIR}/selected_model"
fi

FINETUNED_MODEL="${OUTPUT_DIR}/finetuned_model"

echo "========================================"
echo "HuggingFace Finetuning Pipeline (Continuous-Sparsification Selection)"
echo "========================================"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Selective keep-k: $SELECTIVE_KEEP_K  (shared: $NUM_SHARED_EXPERTS)"
echo "Selection mode: $SELECTION_MODE"
echo "Output dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num epochs: $NUM_EPOCHS"
echo "========================================"

# Per-stage --num-shots forwarding flags. Empty ⇒ task config default.
NUM_SHOTS_SELECTIVE_FLAG=()
if [ -n "$NUM_SHOTS_SELECTIVE" ]; then
    NUM_SHOTS_SELECTIVE_FLAG=(--num-shots "$NUM_SHOTS_SELECTIVE")
fi
NUM_SHOTS_EVAL_FLAG=()
if [ -n "$NUM_SHOTS_EVAL" ]; then
    NUM_SHOTS_EVAL_FLAG=(--num-shots "$NUM_SHOTS_EVAL")
fi

# Steps 1+2: Continuous-Sparsification expert selection
if [ "$SKIP_SELECTIVE" = false ]; then
    echo ""
    echo "Steps 1+2: Continuous-Sparsification expert selection..."
    echo "========================================"

    NUM_CAL_FLAG=()
    if [ -n "$NUM_SELECTIVE_EXAMPLES" ]; then
        NUM_CAL_FLAG=(--num-calibration "$NUM_SELECTIVE_EXAMPLES")
    fi
    SELECTIVE_SEED_FLAG=()
    if [ -n "$NUM_SELECTIVE_SEED" ]; then
        SELECTIVE_SEED_FLAG=(--prune-seed "$NUM_SELECTIVE_SEED")
    fi
    # Optional CS hyperparameter flags — only forwarded when the caller set them,
    # so the python script's own defaults apply otherwise.
    CS_FLAGS=(--selection-mode "$SELECTION_MODE")
    [ -n "$CS_LAMBDA" ]           && CS_FLAGS+=(--lambda "$CS_LAMBDA")
    [ -n "$CS_LR" ]               && CS_FLAGS+=(--lr "$CS_LR")
    [ -n "$CS_NUM_STEPS" ]        && CS_FLAGS+=(--num-steps "$CS_NUM_STEPS")
    [ -n "$CS_BETA_START" ]       && CS_FLAGS+=(--beta-start "$CS_BETA_START")
    [ -n "$CS_BETA_END" ]         && CS_FLAGS+=(--beta-end "$CS_BETA_END")
    [ -n "$CS_S_INIT" ]           && CS_FLAGS+=(--s-init "$CS_S_INIT")
    [ -n "$CS_SELECT_BATCH_SIZE" ] && CS_FLAGS+=(--batch-size "$CS_SELECT_BATCH_SIZE")
    TRC_FLAG=()
    [ "$TRUST_REMOTE_CODE" = true ] && TRC_FLAG=(--trust-remote-code)

    # CS needs masked labels for a task loss → calibrate on the train split.
    python -m src.hf_training.continuous_sparsify_prune \
        --model "$MODEL" \
        --task "$TASK" \
        --split "train" \
        --prune-keep-k "$SELECTIVE_KEEP_K" \
        --num-shared-experts "$NUM_SHARED_EXPERTS" \
        --save-path "$SELECTED_MODEL" \
        "${CS_FLAGS[@]}" \
        "${TRC_FLAG[@]}" \
        "${NUM_CAL_FLAG[@]}" \
        "${SELECTIVE_SEED_FLAG[@]}" \
        "${NUM_SHOTS_SELECTIVE_FLAG[@]}"

    echo "Selected-expert model saved to: $SELECTED_MODEL"
else
    echo ""
    echo "Steps 1+2: Skipping selection (using existing selected model)"
    echo "Selected model: $SELECTED_MODEL"
fi

# Step 3: Finetune
echo ""
echo "Step 3: Finetuning..."
echo "========================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    FSDP_FLAG=""
else
    FSDP_FLAG="--no-fsdp"
fi

export WANDB_PROJECT="olmoe-modular"
export WANDB_ENTITY="ryanyxw"
SM_TAG="${SELECTED_MODEL: -60}"
[ -z "$SM_TAG" ] && SM_TAG="$SELECTED_MODEL"
export WANDB_TAGS="finetune,${TASK:0:60},${SM_TAG}"

gas=$(( BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE) ))

torchrun --nproc_per_node="$NUM_GPUS" \
    -m src.hf_training.finetune \
    --model "$SELECTED_MODEL" \
    --task "$TASK" \
    --split "train" \
    --output-dir "$FINETUNED_MODEL" \
    --num-epochs "$NUM_EPOCHS" \
    --num-checkpoints "$NUM_CHECKPOINTS" \
    --learning-rate "$LEARNING_RATE" \
    --run-name "$RUN_NAME" \
    --per-device-batch-size "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$gas" \
    $FSDP_FLAG \
    "${NUM_SHOTS_EVAL_FLAG[@]}"

# Step 4: Evals
echo ""
echo "Step 4: Evals..."
echo "========================================"

export HF_DATASETS_OFFLINE=1

all_checkpoints=("$FINETUNED_MODEL"/checkpoint-*/)

for checkpoint in "${all_checkpoints[@]}"; do
    echo "Evaluating checkpoint: $checkpoint"
    checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')

    EVAL_BATCH_SIZE=32
    if [[ $TASK == *"history"* ]]; then
      EVAL_BATCH_SIZE=4
    fi
    if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
      EVAL_BATCH_SIZE=16
    fi

    python -m src.scripts.eval.launch_eval \
        --model "$checkpoint" \
        --model-type hf \
        --task "$TASK-pruned" \
        --pruned_split "test" \
        --output-dir "${OUTPUT_DIR}/results/checkpoint-${checkpoint_num}" \
        --batch-size $EVAL_BATCH_SIZE \
        --gpus "$NUM_GPUS" \
        "${NUM_SHOTS_EVAL_FLAG[@]}"
done

# Step 5: Per-subject evals (MMLU only)
MMLU_SUBJECTS=$(python -m src.scripts.eval.get_mmlu_subjects "$TASK" 2>/dev/null | grep -v "^Warning:" || true)

if [ -n "$MMLU_SUBJECTS" ]; then
    echo ""
    echo "Step 5: Per-subject MMLU evals..."
    echo "========================================"

    if [[ $TASK == mmlu_merged_* ]]; then
        SUBJECT_TASK_PREFIX="mmlu_merged_"
    else
        SUBJECT_TASK_PREFIX="mmlu_"
    fi

    for checkpoint in "${all_checkpoints[@]}"; do
        checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')
        echo "Per-subject evals for checkpoint: $checkpoint"

        while IFS= read -r subject; do
            echo "  Evaluating subject: $subject"

            EVAL_BATCH_SIZE=32
            if [[ $subject == *"history"* ]]; then
                EVAL_BATCH_SIZE=4
            fi

            python -m src.scripts.eval.launch_eval \
                --model "$checkpoint" \
                --model-type hf \
                --task "${SUBJECT_TASK_PREFIX}${subject}-pruned" \
                --pruned_split "test" \
                --output-dir "${OUTPUT_DIR}/results/checkpoint-${checkpoint_num}/per_subject/${subject}" \
                --batch-size $EVAL_BATCH_SIZE \
                "${NUM_SHOTS_EVAL_FLAG[@]}" \
                --gpus "$NUM_GPUS"
        done <<< "$MMLU_SUBJECTS"
    done
else
    echo ""
    echo "Skipping per-subject evals (task $TASK is not an MMLU category/cluster)"
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Selected model: $SELECTED_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"
echo "Results: $OUTPUT_DIR/results"
