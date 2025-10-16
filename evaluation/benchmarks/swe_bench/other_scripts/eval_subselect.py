import json
import toml
import argparse
import os
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter JSONL entries based on search criteria.")
    parser.add_argument("--input_file", type=str, help="Path to consolidated json eval")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument('--selected_ids', type=str, required=False, default=None, help="Pass toml file with key selected_ids")
    return parser.parse_args()


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "summary_openhands.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)

def load_data(input_file):
    with open(input_file, "r") as fh:
        data = json.load(fh)

    return data

def filter_data(input_data, selected_ids, output_file):
    output_data = {"selected_ids": selected_ids}
    copy_keys = ["run_id", "predictions_path", "output_directory"]
    for k in copy_keys:
        output_data[k] = input_data[k]

    swe_bench_statistics = input_data["swe_bench_statistics"]
    selected_keys = [
        ("submitted_ids", "instances_submitted"),
        ("completed_ids", "instances_completed"),
        ("incomplete_ids", "instances_incomplete"),
        ("resolved_ids", "instances_resolved"),
        ("unresolved_ids", "instances_unresolved"),
        ("empty_patch_ids", "instances_with_empty_patches"),
        ("error_ids", "instances_with_errors"),
    ]

    output_data["swe_bench_statistics"] = {}
    for (key, tag) in selected_keys:
        output_data["swe_bench_statistics"][key] = [inst for inst in swe_bench_statistics[key] if inst in selected_ids]

    for (key, tag) in selected_keys:
        output_data["swe_bench_statistics"][tag] = len(output_data["swe_bench_statistics"][key])



    detailed_results = []
    total_instances = 0
    successful_loads = 0
    failed_loads = 0
    missing_docker_images = 0
    successful_evals = 0
    skipped_evals = 0
    failed_evals = 0

    for entry in input_data["detailed_results"]:
        if entry["instance_id"] not in selected_ids:
            continue

        detailed_results.append(entry)
        total_instances += 1
        if entry["docker_load_success"]:
            successful_loads += 1
        else:
            failed_loads += 1
            docker_error = entry["docker_load_error"]
            if docker_error and "Docker image tar not found" in docker_error:
                missing_docker_images += 1

        eval_result = entry["evaluation_result"]
        if eval_result:
            if eval_result.get("status") == "success":
                successful_evals += 1
            else:
                failed_evals += 1
        else:
            skipped_evals +=1


    output_data["detailed_results"] = detailed_results
    output_data["swe_bench_statistics"]["total_instances"] = len(detailed_results)


    output_data["execution_statistics"] = {}
    output_data["execution_statistics"]["total_instances"] = total_instances
    output_data["execution_statistics"]["successful_loads"] = successful_loads
    output_data["execution_statistics"]["failed_loads"] = failed_loads
    output_data["execution_statistics"]["missing_docker_images"] = missing_docker_images
    output_data["execution_statistics"]["successful_evals"] = successful_evals
    output_data["execution_statistics"]["failed_evals"] = successful_evals
    output_data["execution_statistics"]["skipped_evals"] = skipped_evals

    output_data["execution_statistics"]["start_time"] = input_data["execution_statistics"]["start_time"]
    output_data["execution_statistics"]["end_time"] = input_data["execution_statistics"]["end_time"]


    with open(output_file, "w") as fh:
        json.dump(output_data, fh, indent=2)

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "summary_openhands.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)

if __name__ == "__main__":

    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file

    logger = setup_logging(os.path.dirname(output_file))
    selected_ids = None
    if args.selected_ids is not None:
        selected_ids = toml.load(args.selected_ids)["selected_ids"]

    logger.info(f"Selected_IDS: {selected_ids}")

    # Run the filtering process
    input_data = load_data(input_file)
    filter_data(input_data, selected_ids, output_file)
