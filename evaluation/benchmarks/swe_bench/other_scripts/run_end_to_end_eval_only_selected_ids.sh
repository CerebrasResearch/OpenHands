#!/bin/bash

#!/bin/bash
set -e  # Exit on any error

##########################
# ----- ARGUMENTS -------#
##########################
EVAL_OUTNAME=$1
LOCAGENT_TOOLS=$2
USE_CMD=$3
TEMPLATE_NAME=$4
ADD_LOCAGENT_TOOLS_FIRST=$5
ALT_LOCAGENT_TOOLS=$6
CODE_COMMENTS_TOOL=$7
THINK_PLAN=$8
STR_REPL_THINK=$9
MODEL=${10:-"llm.qwen_coder_30b_small"}
LOCAL_DOCKER_DIR=${11}
CONFIG_ML=${12}
MAX_TURNS=${13:-100}
NUM_SAMPLES=${14:-200}
DATASET=${15:-"princeton-nlp/SWE-bench"}
SPLIT=${16:-"dev"}
# DATASET="princeton-nlp/SWE-bench_Verified"
# SPLIT="test"

NUM_WORKERS=1
NUM_RUNS=1

#######################################
# 📋 Setup Logging to File and Log CMD
#######################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR_RUN="logs_end_to_end"
mkdir -p "$LOGDIR_RUN"
LOGFILE="$LOGDIR_RUN/eval_run_${EVAL_OUTNAME}_${TIMESTAMP}.log"

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
export EVAL_OUTPUT_DIR="evaluation/$EVAL_OUTNAME/outputs"
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
echo "=========================================================="
echo ""

##########################
# -- RUN INFERENCE ----- #
##########################
# Placeholder for actual inference call
# Uncomment and customize if needed
# /path/to/inference.sh $MODEL ...
#############################
# 🚀 Run Main Evaluation   #
#############################
# echo "🔧 Starting main inference..."
# /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer_local_docker.sh \
#     $MODEL \
#     HEAD \
#     CodeActAgent \
#     $NUM_SAMPLES \
#     $MAX_TURNS \
#     $NUM_WORKERS \
#     $DATASET \
#     $SPLIT \
#     $NUM_RUNS \
#     swe
# echo "✅ Evaluation complete. Results saved in $EVAL_OUTPUT_DIR"
# echo

##########################
# --- TOOL CALL SUMMARY --#
##########################
echo ">>> [1/5] TOOL CALL SUMMARY"

JSONL_DIR="/workspaces/OpenHands/$EVAL_OUTPUT_DIR"
ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl")
JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1)
PARENT_FOLDER=$(dirname "$JSONL_FILE")
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

LOC_SUMMARY_OUTPUT="$PARENT_FOLDER/localization_selected"
mkdir -p "$LOC_SUMMARY_OUTPUT"

python3 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_localisation_report.py \
    --model_name "$MODEL" \
    --predictions_path "$SWEBENCH_JSONL" \
    --report_dir "$LOC_SUMMARY_OUTPUT" \
    --selected_ids "$CONFIG_ML" \
    --dataset_name "$DATASET" \
    --dataset_split "$SPLIT"

echo "✅ Localization summary saved to: $LOC_SUMMARY_OUTPUT"

##########################
# --- FINAL EVAL (TRAJECTORY) ---#
##########################
echo ""
echo ">>> [4/5] TRAJECTORY EVALUATION"

OUT_FINAL="$PARENT_FOLDER/final_eval_selected"
mkdir -p "$OUT_FINAL"
FINAL_PRED_PATH="$SWEBENCH_JSONL"

EXEC_SCRIPT="$PARENT_FOLDER/run_commands.sh"
echo "#!/bin/bash" > "$EXEC_SCRIPT"
echo "" >> "$EXEC_SCRIPT"

cat >> "$EXEC_SCRIPT" << EOF
cd $OUT_FINAL
echo "Current directory: $(pwd)"
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \\
  --run_id "$MODEL" \\
  --predictions_path "$FINAL_PRED_PATH" \\
  --output_dir "$OUT_FINAL" \\
  --dataset_name "$DATASET" \\
  --dataset_split "$SPLIT"  \\
  --selected_ids "$CONFIG_ML"
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
  --dataset_split "$SPLIT" \
  --selected_ids "$CONFIG_ML"

# EVAL_JSON=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json" | head -n 1)
EVAL_JSON=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json" -print0 | xargs -0 ls -t | head -n 1)

echo "✅ Evaluation completed: $EVAL_JSON"

##########################
# --- FILTERED LOCALIZATION ---#
##########################
echo ""
echo ">>> [5/5] FILTERED LOCALIZATION SUMMARY"

TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_filtered_$MODEL_selected_ids"
LOC_JSONL_FILE=$(find "$LOC_SUMMARY_OUTPUT" -type f -name "*localisation_report.jsonl" | head -n 1)

python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT" \
    --loc_json "$LOC_JSONL_FILE" \
    --eval_json "$EVAL_JSON" \
    --selected_ids "$CONFIG_ML"

echo "✅ Filtered localization summary saved."

##########################
# -------- RUN SUMMARY -------- #
##########################
echo ""
echo ">>> GENERATING FINAL RUN SUMMARY"

echo python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \
    --input_file "$JSONL_FILE" \
    --eval_summary_file "$EVAL_JSON" \
    --localization_report "$LOC_JSONL_FILE" \
    --selected_ids "$CONFIG_ML"

python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \
    --input_file "$JSONL_FILE" \
    --eval_summary_file "$EVAL_JSON" \
    --localization_report "$LOC_JSONL_FILE" \
    --selected_ids "$CONFIG_ML"

echo "✅ Final summary generated."

##########################
# -------- DONE -------- #
##########################
echo ""
echo "🎉 All steps completed successfully."
echo "👉 You can re-run evaluation using: $EXEC_SCRIPT"
echo "📂 Results located in: $PARENT_FOLDER"
