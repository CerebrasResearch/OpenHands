#!/bin/bash

#!/bin/bash
set -e  # Exit on any error

##########################
# ----- ARGUMENTS -------#
##########################

LOCAGENT_TOOLS="false"
USE_CMD="true"
TEMPLATE_NAME="swe_default.j2"
ADD_LOCAGENT_TOOLS_FIRST="false"
ALT_LOCAGENT_TOOLS="false"
CODE_COMMENTS_TOOL="false"
THINK_PLAN="false"
STR_REPL_THINK="false"
MODEL="cepov2_optillm_qwen480b_together"
# MODEL="llm.together_qwen_480b"
# LOCAL_DOCKER_DIR="/workspaces/Openhands/swebench_dockers_for_eval/swebench_dockers/dev/docker_images/"
LOCAL_DOCKER_DIR="/workspaces/Openhands/swebench_dockers_for_eval/swebench_verified_dockers/test/docker_images"
CONFIG_ML="/workspaces/OpenHands/evaluation/benchmarks/swe_bench/config_2.toml"
MAX_TURNS=500
# DATASET="princeton-nlp/SWE-bench"
# SPLIT="dev"
DATASET="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
NUM_SAMPLES=10
EVAL_OUTNAME="cepo_michael_v5_1117_qwen480b_together_maxiter_500_N_${NUM_SAMPLES}_verified"


NUM_WORKERS=1
NUM_RUNS=1

EVAL_OUTPUT_DIR="/workspaces/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified/CodeActAgent/$EVAL_OUTNAME/"

# Check if the directory exists and is not empty variable
if [ -n "$EVAL_OUTPUT_DIR" ] && [ -d "$EVAL_OUTPUT_DIR" ]; then
    rm -r "$EVAL_OUTPUT_DIR"
    echo "Removed: $EVAL_OUTPUT_DIR"
else
    echo "Directory does not exist: $EVAL_OUTPUT_DIR"
fi

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
# ---- ENV EXPORTS ------#
##########################
# Usage:
#   ./run.sh                      # default: gen_and_eval
#   ./run.sh inference_only
#   ./run.sh eval_only
#   ./run.sh gen_and_eval
MODE=${1:-gen_and_eval}   # inference_only | eval_only | gen_and_eval

export EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR
export USE_LOCAGENT_TOOLS=$LOCAGENT_TOOLS
export ADD_LOCAGENT_TOOLS_FIRST=$ADD_LOCAGENT_TOOLS_FIRST
export ENABLE_CMD=$USE_CMD
export ALT_LOCAGENT_TOOLS=$ALT_LOCAGENT_TOOLS
export ENABLE_CODE_COMMENTS=$CODE_COMMENTS_TOOL
export THINK_PLAN_BRAINSTORM=$THINK_PLAN
export ENABLE_STR_REPLACE_EDIT_THINK_CHECK=$STR_REPL_THINK
export INSTRUCTION_TEMPLATE_NAME=$TEMPLATE_NAME
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
echo "  EVAL_OUTPUT_DIR:             $EVAL_OUTPUT_DIR"
echo "  MODEL:                       $MODEL"
echo "  USE_LOCAGENT_TOOLS:         $USE_LOCAGENT_TOOLS"
echo "  ADD_LOCAGENT_TOOLS_FIRST:   $ADD_LOCAGENT_TOOLS_FIRST"
echo "  ALT_LOCAGENT_TOOLS:         $ALT_LOCAGENT_TOOLS"
echo "  CODE_COMMENTS_TOOL:         $CODE_COMMENTS_TOOL"
echo "  ENABLE_CMD:                 $ENABLE_CMD"
echo "  THINK_PLAN_BRAINSTORM:      $THINK_PLAN_BRAINSTORM"
echo "  ENABLE_STR_REPLACE:         $ENABLE_STR_REPLACE_EDIT_THINK_CHECK"
echo "  INSTRUCTION_TEMPLATE_NAME:  $INSTRUCTION_TEMPLATE_NAME"
echo "  MAX_TURNS:                  $MAX_TURNS"
echo "  NUM_SAMPLES:                $NUM_SAMPLES"
echo "  NUM_WORKERS:                $NUM_WORKERS"
echo "  NUM_RUNS:                   $NUM_RUNS"
echo "  DATASET:                    $DATASET"
echo "  SPLIT:                      $SPLIT"
echo "  LOCAL_DOCKER_IMAGE_DIR:     $LOCAL_DOCKER_IMAGE_DIR"
echo "  CONFIG_ML:                  $CONFIG_ML"
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
# ---- LOCALIZATION REPORT ----#
##########################
echo ""
echo ">>> [3/5] LOCALIZATION REPORT"

LOC_SUMMARY_OUTPUT="$PARENT_FOLDER/localization"
mkdir -p "$LOC_SUMMARY_OUTPUT"

python3 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_localisation_report.py \
    --model_name "$MODEL" \
    --predictions_path "$SWEBENCH_JSONL" \
    --report_dir "$LOC_SUMMARY_OUTPUT" \
    --dataset_name "$DATASET" \
    --dataset_split "$SPLIT"

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
echo "Current directory: $(pwd)"
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \
  --run_id "$MODEL" \
  --predictions_path "$FINAL_PRED_PATH" \
  --output_dir "$OUT_FINAL" \
  --dataset_name "$DATASET" \
  --dataset_split "$SPLIT"

# EVAL_JSON=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json" | head -n 1)
EVAL_JSON=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json" -print0 | xargs -0 ls -t | head -n 1)


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

echo python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \
    --input_file "$JSONL_FILE" \
    --eval_summary_file "$EVAL_JSON" \
    --localization_report "$LOC_JSONL_FILE"

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
