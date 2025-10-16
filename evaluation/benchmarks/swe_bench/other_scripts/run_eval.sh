
#!/bin/bash


set -e  # Exit on any error

EVAL_OUTNAME=$1
LOCAGENT_TOOLS=$2
USE_CMD=$3
TEMPLATE_NAME=$4
ADD_LOCAGENT_TOOLS_FIRST=$5
ALT_LOCAGENT_TOOLS=$6
CODE_COMMENTS_TOOL=$7
THINK_PLAN=$8
STR_REPL_THINK=$9


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



MODEL="llm.qwen_coder_30b_small"
MAX_TURNS=100
NUM_SAMPLES=200
NUM_WORKERS=1
NUM_RUNS=1
# DATASET="princeton-nlp/SWE-bench_Verified"
# SPLIT="test"
DATASET="princeton-nlp/SWE-bench"
SPLIT="dev"


echo "Running SWE-bench evaluation with:"
echo "  USE_LOCAGENT_TOOLS: $USE_LOCAGENT_TOOLS"
echo "  ADD_LOCAGENT_TOOLS_FIRST: $ADD_LOCAGENT_TOOLS_FIRST"
echo "  ALT_LOCAGENT_TOOLS: $ALT_LOCAGENT_TOOLS"
echo "  CODE_COMMENTS_TOOL: $CODE_COMMENTS_TOOL"
echo "  ENABLE_CMD: $ENABLE_CMD"
echo "  THINK_PLAN_BRAINSTORM: $THINK_PLAN_BRAINSTORM"
echo "  ENABLE_STR_REPLACE_EDIT_THINK_CHECK: $ENABLE_STR_REPLACE_EDIT_THINK_CHECK"
echo "  INSTRUCTION_TEMPLATE_NAME: $INSTRUCTION_TEMPLATE_NAME"
echo "  EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
echo "  MODEL: $MODEL"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo "  NUM_RUNS: $NUM_RUNS"
echo "  DATASET: $DATASET"
echo "  SPLIT: $SPLIT"


# /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
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


# echo "Evaluation completed. Results are saved in $EVAL_OUTPUT_DIR."


## ---------- Post-processing: Summarize tool calls ----------
JSONL_DIR="/workspaces/OpenHands/$EVAL_OUTPUT_DIR"

ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl")

echo "Files in JSONL_DIR:"
echo "$ALL_JSONL_FILES"

## Get the first file path
JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1)
echo "Selected JSONL file: $JSONL_FILE"


PARENT_FOLDER=$(dirname "$JSONL_FILE")
TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_$MODEL"

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT"


echo "Tool call summary saved to $TOOL_SUMMARY_OUTPUT"

## ---------- Post-processing: RUN EVAL ----------

DEBUG=1 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
    $JSONL_FILE \
    "" \
    $DATASET \
    $SPLIT

echo "Final evaluation completed."


## ---------- Post-processing: RUN LOCALIZATION ----------

SWEBENCH_JSONL="$PARENT_FOLDER/output.swebench.jsonl"
LOC_SUMMARY_OUTPUT="$PARENT_FOLDER/localization"

mkdir -p $LOC_SUMMARY_OUTPUT


/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_localisation_report.py \
    --model_name $MODEL \
    --predictions_path $SWEBENCH_JSONL \
    --report_dir $LOC_SUMMARY_OUTPUT \

echo "Localization summary saved to $LOC_SUMMARY_OUTPUT"


## ---------- Post-processing: ECHO final eval script ----------

OUT="$PARENT_FOLDER/final_eval"
mkdir -p $OUT
OUT_FINAL="${OUT//\/workspaces/\/mlf11-shared\/coding\/test}"

FINAL_PRED_PATH="${SWEBENCH_JSONL//\/workspaces/\/mlf11-shared\/coding\/test}"

# Create the execution script file
EXEC_SCRIPT="$PARENT_FOLDER/run_commands.sh"

# Write shebang
echo "#!/bin/bash" > "$EXEC_SCRIPT"
echo "" >> "$EXEC_SCRIPT"

# Echo and write trajectory evaluation command
echo "Running trajectory evaluation:"
cat >> "$EXEC_SCRIPT" << EOF
cd $OUT_FINAL
python /mlf11-shared/coding/test/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh.py \\
  --run_id $MODEL \\
  --predictions_path $FINAL_PRED_PATH \\
  --output_dir $OUT_FINAL

EOF

# Display what was written
cat << EOF
cd $OUT_FINAL
python /mlf11-shared/coding/test/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh.py \\
  --run_id $MODEL \\
  --predictions_path $FINAL_PRED_PATH \\
  --output_dir $OUT_FINAL
EOF

echo -e "\n\n\n"

## ---------- Post-processing: ECHO LOCALIZATION FILTERED ----------

JSONL_DIR="/workspaces/OpenHands/$EVAL_OUTPUT_DIR"

ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl")

echo "Files in JSONL_DIR:"
echo "$ALL_JSONL_FILES"

# Get the first file path
JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1)
echo "Selected JSONL file: $JSONL_FILE"

PARENT_FOLDER=$(dirname "$JSONL_FILE")
TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_filtered_$MODEL"

LOC_JSONLS=$(find "$LOC_SUMMARY_OUTPUT" -type f -name "*localisation_report.jsonl")
LOC_JSONL_FILE=$(echo "$LOC_JSONLS" | head -n 1)

EVAL_JSONLS=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json")
EVAL_JSON=$(echo "$EVAL_JSONLS" | head -n 1)

# Echo and write filtered localization command
echo "Running FILTERED LOCALIZATION:"
cat >> "$EXEC_SCRIPT" << EOF
python \\
    /mlf11-shared/coding/test/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \\
    --input_file "$JSONL_FILE" \\
    --output_dir "$TOOL_SUMMARY_OUTPUT" \\
    --loc_json "$LOC_JSONL_FILE" \\
    --eval_json "$EVAL_JSON"

EOF

# Display what was written
cat << EOF
python \\
    /mlf11-shared/coding/test/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \\
    --input_file "$JSONL_FILE" \\
    --output_dir "$TOOL_SUMMARY_OUTPUT" \\
    --loc_json "$LOC_JSONL_FILE" \\
    --eval_json "$EVAL_JSON"
EOF

echo -e "\n\n\n"

# Make the script executable
chmod +x "$EXEC_SCRIPT"

echo "Commands written to: $EXEC_SCRIPT"
echo "To execute, run: $EXEC_SCRIPT"





# # # ------------------------ PREVIOUS ------------------------
# # # ---------- Post-processing: ECHO final eval script ----------

# OUT="$PARENT_FOLDER/final_eval"
# mkdir -p $OUT
# OUT_FINAL="${OUT//\/workspaces/\/mlf11-shared\/coding\/test}"


# FINAL_PRED_PATH="${SWEBENCH_JSONL//\/workspaces/\/mlf11-shared\/coding\/test}"

# echo "Running trajectory evaluation:"
# echo "python /mlf11-shared/coding/test/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh.py \\"
# echo "  --run_id $MODEL \\"
# echo "  --predictions_path  $FINAL_PRED_PATH \\"
# echo "  --output_dir $OUT_FINAL"
# echo -e "\n\n\n"

# # # ---------- Post-processing: ECHO LOCALIZATION FILTERED ----------

# JSONL_DIR="/workspaces/OpenHands/$EVAL_OUTPUT_DIR"

# ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl")

# echo "Files in JSONL_DIR:"
# echo "$ALL_JSONL_FILES"

# # Get the first file path
# JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1)
# echo "Selected JSONL file: $JSONL_FILE"


# PARENT_FOLDER=$(dirname "$JSONL_FILE")
# TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_filtered_$MODEL"

# LOC_JSONLS=$(find "$LOC_SUMMARY_OUTPUT" -type f -name "*localisation_report.jsonl")
# LOC_JSONL_FILE=$(echo "$LOC_JSONLS" | head -n 1)

# EVAL_JSONLS=$(find "$OUT_FINAL" -type f -name "consolidated_report*.json")
# EVAL_JSON=$(echo "$EVAL_JSONLS" | head -n 1)

# echo "Running FILTERED LOCALIZATION:"
# echo "/python \\
#     /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \\
#     --input_file \"$JSONL_FILE\" \\
#     --output_dir \"$TOOL_SUMMARY_OUTPUT\" \\
#     --loc_json \"$LOC_JSONL_FILE\" \\
#     --eval_json \"$EVAL_JSON\""

# echo -e "\n\n\n"


# # # ------------------------ END PREVIOUS ------------------------
