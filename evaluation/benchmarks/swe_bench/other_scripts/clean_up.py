

import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean up unnecessary files after OpenHands Swe-bench evaluation")
    parser.add_argument("--evaluation_folder", type=str, help="A folder like /workspaces/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_176_baseline")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    base_folder = args.evaluation_folder

    llm_completion_folder = os.path.join(base_folder, "llm_completions")


    instance_ids = [d for d in os.listdir(llm_completion_folder) if os.path.isdir(os.path.join(llm_completion_folder, d))]


    for instance_id in instance_ids:

        instance_folder = os.path.join(llm_completion_folder, instance_id)

        intermediate_files = sorted(
            [f for f in os.listdir(instance_folder) if os.path.isfile(os.path.join(instance_folder, f)) and "trajectory" not in f]
        )

        # Skip the final messages json file, delete only the previous ones
        for file in intermediate_files[:-1]:
            os.remove(os.path.join(instance_folder, file))
