import os
import asyncio
from typing import Optional, Iterable

import pandas as pd
from datasets import load_dataset

from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.config.condenser_config import NoOpCondenserConfig

from evaluation.benchmarks.swe_bench.run_infer_local_docker import (
    set_dataset_type,
    get_config,
    get_instruction,
    initialize_runtime,
    complete_runtime,
)

from evaluation.utils.shared import (
    make_metadata,
    get_metrics,
    codeact_user_response,
    prepare_dataset,
)

from openhands.core.config import (
    get_llm_config_arg,
)

# ============================================================
# 0. ENV + HIGH-LEVEL CONFIG
# ============================================================

# Path to directory with your SWE-bench docker .tar images
os.environ["LOCAL_DOCKER_IMAGE_DIR"] = (
    "/workspaces/Openhands/swebench_dockers_for_eval/"
    "swebench_verified_dockers/test/docker_images"
)

# LLM / eval config
LLM_CONFIG_NAME = "llm.together_qwen_480b"     # must exist in config.toml
CONFIG_FILE = "config.toml"

DATASET_NAME, SPLIT = "princeton-nlp/SWE-bench_Verified", "test"
AGENT_CLS = "CodeActAgent"
MAX_ITERATIONS = 50
EVAL_OUTPUT_DIR = "qwen480b_single_file_inference_test"
EVAL_NOTE = None
MODE = "swe"  # 'swe', 'swt', or 'swt-ci'

# Instance filter
SELECTED_IDS = [
    "django__django-14792",
    "astropy__astropy-14365",
    "mwaskom__seaborn-3187",
    "matplotlib__matplotlib-20676",
    "psf__requests-2931",
    "sphinx-doc__sphinx-7454",
    "pylint-dev__pylint-6386",
    "pydata__xarray-3993",
    "scikit-learn__scikit-learn-25747",
    "sympy__sympy-15976",
]


# ============================================================
# 1. HELPERS
# ============================================================

def cleanup_extra_trajectories(eval_folder):
    llm_completion_folder = os.path.join(eval_folder, "llm_completions")
    instance_ids = [d for d in os.listdir(llm_completion_folder) if os.path.isdir(os.path.join(llm_completion_folder, d))]
    for instance_id in instance_ids:

        instance_folder = os.path.join(llm_completion_folder, instance_id)

        intermediate_files = sorted(
            [f for f in os.listdir(instance_folder) if os.path.isfile(os.path.join(instance_folder, f)) and "trajectory" not in f]
        )

        # Skip the final messages json file, delete only the previous ones
        for file in intermediate_files[:-1]:
            os.remove(os.path.join(instance_folder, file))

def filter_dataset(
    dataset: pd.DataFrame,
    filter_column: str,
    selected_ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Filter `dataset` to rows where `filter_column` is in `selected_ids`.

    If `selected_ids` is None, return the original dataset unchanged.
    """
    if selected_ids is None:
        print("No selected_ids provided; returning full dataset")
        return dataset

    subset = dataset[dataset[filter_column].isin(selected_ids)]
    print(
        f"Filtered dataset on {filter_column} using {len(selected_ids)} selected_ids; "
        f"retained {subset.shape[0]} / {dataset.shape[0]} rows"
    )
    return subset


# ============================================================
# 2. MAIN SINGLE-INSTANCE FLOW
# ============================================================

def main():
    # -------------------------
    # 2.1 LLM config + metadata
    # -------------------------
    llm_config = get_llm_config_arg(LLM_CONFIG_NAME, CONFIG_FILE)
    llm_config.log_completions = True
    llm_config.modify_params = False  # for reproducibility

    condenser_name = os.environ.get("EVAL_CONDENSER")
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, CONFIG_FILE)
        if condenser_config is None:
            raise ValueError(
                f"Could not find Condenser config: EVAL_CONDENSER={condenser_name}"
            )
    else:
        condenser_config = NoOpCondenserConfig()
        print("No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.")

    dataset_description = DATASET_NAME.replace("/", "__") + "-" + SPLIT.replace("/", "__")
    details = {"mode": MODE}

    # ⚠️ Use the SAME style of make_metadata as run_infer_local_docker
    metadata = make_metadata(
        llm_config,
        dataset_description,
        AGENT_CLS,
        MAX_ITERATIONS,
        EVAL_NOTE,
        EVAL_OUTPUT_DIR,
        details=details,
        agent_config=None,
        condenser_config=condenser_config,
    )

    # Use your custom prompt template
    metadata.instruction_template_name = "swe_michael.j2"

    # -------------------------
    # 2.2 Dataset loading + filtering
    # -------------------------
    print(f"Loading dataset {DATASET_NAME} [{SPLIT}]...")
    hf_ds = load_dataset(DATASET_NAME, split=SPLIT)

    # sets global DATASET_TYPE, affects docker image choice, workspace name, etc.
    set_dataset_type(DATASET_NAME)

    swe_bench_tests = filter_dataset(hf_ds.to_pandas(), "instance_id", selected_ids=SELECTED_IDS)
    print(f"Loaded {len(swe_bench_tests)} filtered tasks")

    if len(swe_bench_tests) == 0:
        raise RuntimeError("No tasks left after filtering; check SELECTED_IDS.")

    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    print(f"### OUTPUT FILE: {output_file} ###")

    eval_n_limit = 1  # only run one instance for now
    instances = prepare_dataset(swe_bench_tests, output_file, eval_n_limit)

    # PASS_TO_PASS / FAIL_TO_PASS columns formatting, same as original
    if len(instances) > 0 and not isinstance(
        instances["PASS_TO_PASS"][instances["PASS_TO_PASS"].index[0]], str
    ):
        for col in ["PASS_TO_PASS", "FAIL_TO_PASS"]:
            instances[col] = instances[col].apply(lambda x: str(x))

    print("Instances prepared:")
    print(instances.head())

    # Take the first instance
    instance = instances.iloc[0]
    print(f"\nRunning single instance: {instance['instance_id']}")

    # -------------------------
    # 2.3 Build OpenHands config + runtime
    # -------------------------
    config = get_config(instance, metadata)

    runtime = create_runtime(config)
    asyncio.run(runtime.connect())

    try:
        # -------------------------
        # 2.4 Initialize runtime (repo, env, etc.)
        # -------------------------
        initialize_runtime(runtime, instance, metadata)

        # -------------------------
        # 2.5 Build initial user message (prompt)
        # -------------------------
        message_action = get_instruction(instance, metadata)
        print("\n===== INITIAL PROMPT =====\n")
        print(message_action.content)

        # -------------------------
        # 2.6 Run controller (CodeActAgent lives here)
        # -------------------------
        state = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,  # CodeActAgent only
            )
        )

        if state is None:
            raise RuntimeError("State is None after run_controller")

        # -------------------------
        # 2.7 Extract git patch
        # -------------------------
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val["git_patch"]

        print("\n===== FINAL GIT PATCH =====\n")
        print(git_patch)

        # -------------------------
        # 2.8 Metrics (optional)
        # -------------------------
        metrics = get_metrics(state)
        print("\n===== METRICS =====\n")
        print(metrics)

        cleanup_extra_trajectories(metadata.eval_output_dir)

    finally:
        runtime.close()


if __name__ == "__main__":
    main()
