
#!/bin/bash


set -e  # Exit on any error

EVAL_OUTNAME=$1
LOCAGENT_TOOLS=$2
USE_CMD=$3
TEMPLATE_NAME=$4
ADD_LOCAGENT_TOOLS_FIRST=$5
ALT_LOCAGENT_TOOLS=$6


export EVAL_OUTPUT_DIR="evaluation/$EVAL_OUTNAME/outputs"
export USE_LOCAGENT_TOOLS=$LOCAGENT_TOOLS
export ADD_LOCAGENT_TOOLS_FIRST=$ADD_LOCAGENT_TOOLS_FIRST
export ENABLE_CMD=$USE_CMD
export ALT_LOCAGENT_TOOLS=$ALT_LOCAGENT_TOOLS

export INSTRUCTION_TEMPLATE_NAME=$TEMPLATE_NAME
export DEBUG=1



MODEL="llm.qwen_coder_30b_small"
MAX_TURNS=100
NUM_SAMPLES=50
NUM_WORKERS=1
NUM_RUNS=1
DATASET="princeton-nlp/SWE-bench_Verified"
SPLIT="test"


echo "Running SWE-bench evaluation with:"
echo "  USE_LOCAGENT_TOOLS: $USE_LOCAGENT_TOOLS"
echo "  ADD_LOCAGENT_TOOLS_FIRST: $ADD_LOCAGENT_TOOLS_FIRST"
echo "  ALT_LOCAGENT_TOOLS: $ALT_LOCAGENT_TOOLS"
echo "  ENABLE_CMD: $ENABLE_CMD"
echo "  INSTRUCTION_TEMPLATE_NAME: $INSTRUCTION_TEMPLATE_NAME"
echo "  EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
echo "  MODEL: $MODEL"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo "  NUM_RUNS: $NUM_RUNS"
echo "  DATASET: $DATASET"
echo "  SPLIT: $SPLIT"


/workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
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


echo "Evaluation completed. Results are saved in $EVAL_OUTPUT_DIR."




# ---------- Post-processing: Summarize tool calls ----------
JSONL_DIR="/workspaces/OpenHands/$EVAL_OUTPUT_DIR"

ALL_JSONL_FILES=$(find "$JSONL_DIR" -type f -name "output.jsonl")

echo "Files in JSONL_DIR:"
echo "$ALL_JSONL_FILES"

# Get the first file path
JSONL_FILE=$(echo "$ALL_JSONL_FILES" | head -n 1)
echo "Selected JSONL file: $JSONL_FILE"


PARENT_FOLDER=$(dirname "$JSONL_FILE")
TOOL_SUMMARY_OUTPUT="$PARENT_FOLDER/bash_tool_call_summary_$MODEL"

/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/bash_tool_call_summary.py \
    --input_file "$JSONL_FILE" \
    --output_dir "$TOOL_SUMMARY_OUTPUT"


echo "Tool call summary saved to $TOOL_SUMMARY_OUTPUT"

# ---------- Post-processing: RUN EVAL ----------

DEBUG=1 /workspaces/OpenHands/evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
    $JSONL_FILE \
    "" \
    $DATASET \
    $SPLIT

echo "Final evaluation completed."


# ---------- Post-processing: RUN LOCALIZATION ----------

SWEBENCH_JSONL="$PARENT_FOLDER/output.swebench.jsonl"
LOC_SUMMARY_OUTPUT="$PARENT_FOLDER/localization"

mkdir -p $LOC_SUMMARY_OUTPUT


/home/vscode/.cache/pypoetry/virtualenvs/openhands-ai-QLt0qIPP-py3.12/bin/python \
    /workspaces/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/generate_localisation_report.py \
    --model_name $MODEL \
    --predictions_path $SWEBENCH_JSONL \
    --report_dir $LOC_SUMMARY_OUTPUT \

echo "Localization summary saved to $LOC_SUMMARY_OUTPUT"


# ---------- Post-processing: ECHO final eval script ----------
OUT="$PARENT_FOLDER/final_eval"
mkdir -p $OUT
OUT_FINAL="${OUT//\/workspaces/\/mlf11-shared\/coding\/aarti_oh_2}"


FINAL_PRED_PATH="${SWEBENCH_JSONL//\/workspaces/\/mlf11-shared\/coding\/aarti_oh_2}"

echo "Running trajectory evaluation:"
echo "python /mlf11-shared/coding/aarti_oh_2/OpenHands/evaluation/benchmarks/swe_bench/other_scripts/evaluate_trajectory_harsh.py \\"
echo "  --run_id $MODEL \\"
echo "  --predictions_path  $FINAL_PRED_PATH \\"
echo "  --output_dir $OUT_FINAL"
