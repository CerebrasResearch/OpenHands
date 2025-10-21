
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
MODEL=${10:-"llm.qwen_coder_30b_small"}


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



# MODEL="llm.cerebras_qwen_480b"
# MODEL="llm.together_qwen_480b"

MAX_TURNS=${11:-100}
NUM_SAMPLES=${12:-200}
DATASET=${13:-"princeton-nlp/SWE-bench"}
SPLIT=${14:-"dev"}
# DATASET="princeton-nlp/SWE-bench_Verified"
# SPLIT="test"

NUM_WORKERS=1
NUM_RUNS=1




echo "Running SWE-bench evaluation with:"
echo "------------------------------------------------------"
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
echo "------------------------------------------------------"



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

echo "-----------------------------"

echo "/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary.py \
    --input_file \"$JSONL_FILE\" \
    --output_dir \"$TOOL_SUMMARY_OUTPUT\""

echo "-----------------------------"

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT"


echo "Tool call summary saved to $TOOL_SUMMARY_OUTPUT"

# ## ---------- Post-processing: RUN EVAL ----------

# DEBUG=1 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
#     $JSONL_FILE \
#     "" \
#     $DATASET \
#     $SPLIT

# echo "Final evaluation completed."

#### ------------ Post processing - generate output.swebench.jsonl ------------

# SWE-bench format is a JSONL where every line has three fields: model_name_or_path, instance_id, and model_patch
function is_swebench_format() {
    # Read the first line of the file
    read -r first_line < "$JSONL_FILE"

    # Use jq to check if the first line has the required fields
    echo "$first_line" | jq -e '. | has("model_name_or_path") and has("instance_id") and has("model_patch")' > /dev/null

    if [ $? -ne 0 ]; then
        return 1 # Return 1 if the first line does not have the required fields
    fi

    return 0 # Return 0 if the first line has the required fields
}

# Call the function with the file path
is_swebench_format "$JSONL_FILE"
FILE_DIR=$(dirname $JSONL_FILE)
IS_SWEBENCH_FORMAT=$?
# Use the result in an if-else statement
if [ $IS_SWEBENCH_FORMAT -eq 0 ]; then
    echo "The file IS in SWE-bench format."
    SWEBENCH_FORMAT_JSONL=$JSONL_FILE
else
    echo "The file IS NOT in SWE-bench format."

    # ==== Convert OH format to SWE-bench format ====
    echo "Merged output file with fine-grained report will be saved to $FILE_DIR"
    poetry run python3 evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py $PROCESS_FILEPATH
    # replace .jsonl with .swebench.jsonl in filename
    SWEBENCH_FORMAT_JSONL=${JSONL_FILE/.jsonl/.swebench.jsonl}
    echo "SWEBENCH_FORMAT_JSONL: $SWEBENCH_FORMAT_JSONL"
    # assert that the file exists
    if [ ! -f $SWEBENCH_FORMAT_JSONL ]; then
        echo "Error: $SWEBENCH_FORMAT_JSONL does not exist. There is probably an error in the conversion process."
        exit 1
    fi
    SWEBENCH_FORMAT_JSONL=$(realpath $SWEBENCH_FORMAT_JSONL)
fi
# ================================================


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
OUT_FINAL="${OUT}"

FINAL_PRED_PATH="$SWEBENCH_JSONL"

# Create the execution script file
EXEC_SCRIPT="$PARENT_FOLDER/run_commands.sh"

# Write shebang
echo "#!/bin/bash" > "$EXEC_SCRIPT"
echo "" >> "$EXEC_SCRIPT"

# Echo and write trajectory evaluation command
echo "Running trajectory evaluation:"
cat >> "$EXEC_SCRIPT" << EOF
cd $OUT_FINAL
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \\
  --run_id $MODEL \\
  --predictions_path $FINAL_PRED_PATH \\
  --output_dir $OUT_FINAL \\
  --dataset_name $DATASET \\
  --dataset_split $SPLIT

EOF

# Display what was written
cat << EOF
cd $OUT_FINAL
python /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \\
  --run_id $MODEL \\
  --predictions_path $FINAL_PRED_PATH \\
  --output_dir $OUT_FINAL
  --dataset_name $DATASET \\
  --dataset_split $SPLIT
EOF

echo -e "\n\n\n"

cd $OUT_FINAL

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh_local.py \
    --run_id $MODEL \
    --predictions_path $FINAL_PRED_PATH \
    --output_dir $OUT_FINAL \
    --dataset_name $DATASET \
    --dataset_split $SPLIT

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
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \\
    --input_file "$JSONL_FILE" \\
    --output_dir "$TOOL_SUMMARY_OUTPUT" \\
    --loc_json "$LOC_JSONL_FILE" \\
    --eval_json "$EVAL_JSON"

EOF

# Display what was written
cat << EOF
python \\
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \\
    --input_file "$JSONL_FILE" \\
    --output_dir "$TOOL_SUMMARY_OUTPUT" \\
    --loc_json "$LOC_JSONL_FILE" \\
    --eval_json "$EVAL_JSON"
EOF

echo -e "\n\n\n"

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary_filtered.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT" \
    --loc_json "$LOC_JSONL_FILE" \
    --eval_json "$EVAL_JSON"

# Make the script executable
chmod +x "$EXEC_SCRIPT"

echo "Commands written to: $EXEC_SCRIPT"
echo "To execute, run: $EXEC_SCRIPT"

#### -------------------SUMMARY--------------------

echo "python \\
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \\
    --input_file \"$JSONL_FILE\" \\
    --eval_summary_file \"$EVAL_JSON\" \\
    --localization_report \"$LOC_JSONL_FILE\""

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_run_summary.py \
    --input_file "$JSONL_FILE" \
    --eval_summary_file "$EVAL_JSON" \
    --localization_report "$LOC_JSONL_FILE"

echo "Summary report generated."

cd /workspaces/OpenHands

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
