#!/bin/bash
set -e  # Exit on any error

#######################################
# 📋 Setup Logging to File and Log CMD
#######################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOGFILE="eval_run_${TIMESTAMP}.log"

# Start logging stdout and stderr
exec > >(tee -i "$LOGFILE") 2>&1

# Log the command that started the script
echo "🔧 Command:"
echo "$0 $@"
echo "========================================"
echo "📅 Started at: $(date)"
echo "📁 Log file: $LOGFILE"
echo "========================================"


##########################
# ----- ARGUMENTS -------#
##########################
# Usage:
#   ./run.sh                      # default: gen_and_eval
#   ./run.sh inference_only
#   ./run.sh eval_only
#   ./run.sh gen_and_eval
MODE=${1:-gen_and_eval}   # inference_only | eval_only | gen_and_eval

EVAL_OUTPUT_DIR="/workspaces/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/cepov3_optillm_qwen30b_maxiter_100_N_50_baseline/"
# MODEL="llm.qwen_coder_30b_small"
MODEL="llm.cepov3_optillm_qwen30b"
LOCAL_DOCKER_DIR="/workspaces/Openhands/swebench_dockers_for_eval/swebench_dockers/dev/docker_images/"
CONFIG_ML="/workspaces/OpenHands/evaluation/benchmarks/swe_bench/config_2.toml"
MAX_TURNS=100
NUM_SAMPLES=1
DATASET="princeton-nlp/SWE-bench"
SPLIT="dev"
# DATASET="princeton-nlp/SWE-bench_Verified"
# SPLIT="test"

NUM_WORKERS=1
NUM_RUNS=1

##########################
# ---- ENV EXPORTS ------#
##########################
export EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR
export DEBUG=1
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export LOCAL_DOCKER_IMAGE_DIR=$LOCAL_DOCKER_DIR
export CONFIG_ML=$CONFIG_ML
export OPENAI_API_KEY="serving-on-vllm"

##########################
# ----- CONFIG LOG ------#
##########################
echo ""
echo "=========== SWE-Bench Evaluation Configuration ==========="
echo "  MODE:                       $MODE"
echo "  EVAL_OUTPUT_DIR:            $EVAL_OUTPUT_DIR"
echo "  MODEL:                      $MODEL"
echo "  MAX_TURNS:                  $MAX_TURNS"
echo "  NUM_SAMPLES:                $NUM_SAMPLES"
echo "  NUM_WORKERS:                $NUM_WORKERS"
echo "  NUM_RUNS:                   $NUM_RUNS"
echo "  DATASET:                    $DATASET"
echo "  SPLIT:                      $SPLIT"
echo "  LOCAL_DOCKER_IMAGE_DIR:     $LOCAL_DOCKER_IMAGE_DIR"
echo "=========================================================="
echo ""

##########################
# -- RUN INFERENCE ----- #
##########################
JSONL_DIR=$EVAL_OUTPUT_DIR

if [[ "$MODE" == "inference_only" || "$MODE" == "gen_and_eval" ]]; then
    echo "🔧 Starting main inference..."
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer_local_docker.sh \
        $MODEL \
        HEAD \
        CodeActAgent \
        $NUM_SAMPLES \
        $MAX_TURNS \
        $NUM_WORKERS \
        $DATASET \
        $SPLIT \
        $NUM_RUNS \
        swe
    echo "✅ Inference complete. Results saved in $EVAL_OUTPUT_DIR"
    echo
else
    echo "⏭️  Skipping inference step (mode=$MODE)"
fi

############################################
# From here on: steps that need output.jsonl
############################################

# Find the produced JSONL (or the existing one, if eval_only)
ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl" || true)
JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1 || true)

if [[ -z "$JSONL_FILE" ]]; then
    echo "❌ Could not find output.jsonl under $JSONL_DIR"
    echo "   Make sure inference has been run or EVAL_OUTPUT_DIR is correct."
    exit 1
fi

PARENT_FOLDER=$(dirname "$JSONL_FILE")

# If mode is inference_only, stop here
if [[ "$MODE" == "inference_only" ]]; then
    echo "🎉 inference_only mode: stopping after inference."
    echo "📂 Results located in: $PARENT_FOLDER"

    # inference_only: do cleanup before exiting
    echo ""
    echo ">>> CLEANUP EXTRA MODEL MESSAGE HISTORY FILES (inference_only)"
    python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/clean_up.py \
        --evaluation_folder "$EVAL_OUTPUT_DIR"
    echo "✅ Cleaned up all intermediate model generation files (inference_only)"

    exit 0
fi

#################################
# Below runs for: gen_and_eval, eval_only
#################################

##########################
# --- TOOL CALL SUMMARY --#
##########################
echo ">>> [1/5] TOOL CALL SUMMARY"

TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_$MODEL"

echo "    Using JSONL file: $JSONL_FILE"
echo "    Saving summary to: $TOOL_SUMMARY_OUTPUT"

python3 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT"

echo "✅ Tool call summary complete."

##########################
# --- CONVERT TO SWE-BENCH FORMAT --#
##########################
echo ""
echo ">>> [2/5] CONVERTING OUTPUT TO SWE-BENCH FORMAT"

python3 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py "$JSONL_FILE"

SWEBENCH_JSONL="$PARENT_FOLDER/output.swebench.jsonl"
echo "✅ Converted to: $SWEBENCH_JSONL"

##########################
# ---- LOCALization REPORT ----#
##########################
echo ""
echo ">>> [3/5] LOCALIZATION REPORT"

LOC_SUMMARY_OUTPUT="$PARENT_FOLDER/localization"
mkdir -p "$LOC_SUMMARY_OUTPUT"

python3 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_localisation_report.py \
    --model_name "$MODEL" \
    --predictions_path "$SWEBENCH_JSONL" \
    --report_dir "$LOC_SUMMARY_OUTPUT"

echo "✅ Localization summary saved to: $LOC_SUMMARY_OUTPUT"

##########################
# --- FINAL EVAL (TRAJECTORY) ---#
##########################
echo ""
echo ">>> [4/5] TRAJECTORY EVALUATION"

OUT_FINAL="$PARENT_FOLDER/final_eval"
mkdir -p "$OUT_FINAL"
FINAL_PRED_PATH="$SWEBENCH_JSONL"

EXEC_SCRIPT="$PARENT_FOLDER/run_commands.sh"
echo "#!/bin/bash" > "$EXEC_SCRIPT"
echo "" >> "$EXEC_SCRIPT"

cat >> "$EXEC_SCRIPT" << EOF
cd $OUT_FINAL
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \\
  --run_id "$MODEL" \\
  --predictions_path "$FINAL_PRED_PATH" \\
  --output_dir "$OUT_FINAL" \\
  --dataset_name "$DATASET" \\
  --dataset_split "$SPLIT"
EOF

chmod +x "$EXEC_SCRIPT"
cat "$EXEC_SCRIPT"

# Run it now
cd "$OUT_FINAL"
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \
  --run_id "$MODEL" \
  --predictions_path "$FINAL_PRED_PATH" \
  --output_dir "$OUT_FINAL" \
  --dataset_name "$DATASET" \
  --dataset_split "$SPLIT"

EVAL_JSON=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json" | head -n 1)

echo "✅ Evaluation completed: $EVAL_JSON"

##########################
# --- FILTERED LOCALIZATION ---#
##########################
echo ""
echo ">>> [5/5] FILTERED LOCALIZATION SUMMARY"

TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_filtered_$MODEL"
LOC_JSONL_FILE=$(find "$LOC_SUMMARY_OUTPUT" -type f -name "*localisation_report.jsonl" | head -n 1)

python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT" \
    --loc_json "$LOC_JSONL_FILE" \
    --eval_json "$EVAL_JSON"

echo "✅ Filtered localization summary saved."

##########################
# -------- RUN SUMMARY -------- #
##########################
echo ""
echo ">>> GENERATING FINAL RUN SUMMARY"

python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \
    --input_file "$JSONL_FILE" \
    --eval_summary_file "$EVAL_JSON" \
    --localization_report "$LOC_JSONL_FILE"

echo "✅ Final summary generated."

##########################
# -------- CLEAN UP -------- #
##########################
echo ""
echo ">>> CLEANUP EXTRA MODEL MESSAGE HISTORY FILES"
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/clean_up.py \
    --evaluation_folder "$EVAL_OUTPUT_DIR"
echo "✅ Cleaned up all intermediate model generation files"

##########################
# -------- DONE -------- #
##########################
echo ""
echo "🎉 All steps completed successfully."
echo "👉 You can re-run evaluation using: $EXEC_SCRIPT"
echo "📂 Results located in: $PARENT_FOLDER"
