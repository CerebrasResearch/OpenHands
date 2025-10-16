import json
from collections import defaultdict



LIST_OF_SUMMARY = {
    "Baseline": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_baseline_cmd_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",
    "LOCPr5_Qwen30B": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",
    "LOCPr5_str_replace_edit_think_plan_brainstorm_Qwen30B": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_str_replace_think_plan_50inst/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json"
}

BASELINE = "Baseline"


def load_json(json_path):
    with open(json_path, 'r') as fh:
        data = json.load(fh)
    return data


def aggregate_metrics(entries):
    """
    Aggregate metrics from a list of entries.

    Args:
        entries: List of dictionaries containing entry data

    Returns:
        Dictionary with aggregated metrics
    """
    total_tool_calls = defaultdict(int)
    total_bash_calls = defaultdict(int)
    total_resolved = 0
    total_llm_metrics = {
        "num_calls_to_llm": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    total_llm_metrics_resolved = {
        "num_calls_to_llm": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    total_localization_stats = {
        "precision": 0,
        "recall": 0
    }
    num_aggregated = {"precision": 0.0, "recall": 0.0}

    num_entries = len(entries)

    for entry in entries:
        # Aggregate tool call counts
        if "tool_call_counts" in entry:
            for tool, count in entry["tool_call_counts"].items():
                total_tool_calls[tool] += count

        # Aggregate bash tool call counts
        if "bash_tool_call_counts" in entry:
            for bash_cmd, count in entry["bash_tool_call_counts"].items():
                total_bash_calls[bash_cmd] += count

        # Aggregate LLM metrics
        if "llm_metrics" in entry:
            for metric, value in entry["llm_metrics"].items():
                if metric in total_llm_metrics:
                    total_llm_metrics[metric] += value

        total_resolved += int(entry["is_resolved"])

        # Aggregate LLM metrics

        for key in ["precision", "recall"]:
            if key in entry:
                total_localization_stats[key] += entry[key]
                num_aggregated[key] += 1

        # print(f"total_localization_stats:{total_localization_stats}")
        # print(f"num_aggregated:{num_aggregated}")

    for key in ["precision", "recall"]:
        total_localization_stats[key] = total_localization_stats[key] / num_aggregated[key] if num_aggregated[key] else 0

    return {
        "num_entries": num_entries,
        "total_resolved": total_resolved,
        "total_tool_calls": dict(total_tool_calls),
        "total_bash_tool_calls": dict(total_bash_calls),
        "total_llm_metrics": total_llm_metrics,
        "localization_avg_metrics": dict(total_localization_stats),
        "grand_total_tool_calls": sum(total_tool_calls.values()),
        "grand_total_bash_calls": sum(total_bash_calls.values()),
        "avg_llm_calls_per_instance": total_llm_metrics["num_calls_to_llm"]/num_entries if num_entries else 0,
        "avg_prompt_tokens_per_instance": total_llm_metrics["prompt_tokens"]/num_entries if num_entries else 0,
        "avg_completion_tokens_tokens_per_instance": total_llm_metrics["completion_tokens"]/num_entries if num_entries else 0,

    }


def _missing_resolved():
    all_data = {}
    for name, path in LIST_OF_SUMMARY.items():
        try:
            data = load_json(path)
            all_data[name] = data
            print(f"Loaded {name}: {len(data)} entries")
        except FileNotFoundError:
            print(f"Warning: Could not find file for {name}: {path}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    resolved = dict()
    for k, entries in all_data.items():
        resolved[k] = []
        for entry in entries:
            if entry["is_resolved"]:
                resolved[k].append(entry["instance_id"])


    union_instance = set(resolved["Baseline"]).union(set(resolved["LOCPr5_Qwen30B"]))

    ran_already = set([entry["instance_id"] for entry in all_data["LOCPr5_str_replace_edit_think_plan_brainstorm_Qwen30B"]])

    to_run =  union_instance.difference(ran_already)

    for x in to_run:
        print(x)


def filter_data(baseline_data, compare_data, filter_type="resolved", key1="baseline", key2="compare"):
    if filter_type == "resolved":

        ### individual
        baseline_resolved_ids = {entry["instance_id"]: entry for entry in baseline_data if entry["is_resolved"]}
        compare_data_resolved_ids = {entry["instance_id"]: entry for entry in compare_data if entry["is_resolved"]}

        baseline_resolved_aggregate_metrics = aggregate_metrics(list(baseline_resolved_ids.values()))
        compare_resolved_aggregate_metrics = aggregate_metrics(list(compare_data_resolved_ids.values()))

        #### union
        union = list(set(baseline_resolved_ids.keys()).union(set(compare_data_resolved_ids.keys())))
        union_resolved_id_entries_in_baseline = [entry for entry in baseline_data if entry["instance_id"] in union]
        union_resolved_id_entries_in_compare = [entry for entry in compare_data if entry["instance_id"] in union]

        resolved_union_aggregate_metrics_in_baseline = aggregate_metrics(union_resolved_id_entries_in_baseline)
        resolved_union_aggregate_metrics_in_compare = aggregate_metrics(union_resolved_id_entries_in_compare)


        #### intersection
        intersection = list(set(baseline_resolved_ids.keys()).intersection(set(compare_data_resolved_ids.keys())))
        intersection_resolved_id_entries_in_baseline = [entry for entry in baseline_data if entry["instance_id"] in intersection]
        intersection_resolved_id_entries_in_compare = [entry for entry in compare_data if entry["instance_id"] in intersection]

        resolved_intersection_aggregate_metrics_in_baseline = aggregate_metrics(intersection_resolved_id_entries_in_baseline)
        resolved_intersection_aggregate_metrics_in_compare = aggregate_metrics(intersection_resolved_id_entries_in_compare)

    elif filter_type == "recall_eq_1":
        baseline_recall_eq1_ids = {entry["instance_id"]: entry for entry in baseline_data if entry["recall"] == 1.0}
        compare_data_recall_eq1_ids = {entry["instance_id"]: entry for entry in compare_data if entry["recall"] == 1.0}

        baseline_recall_eq1_aggregate_metrics = aggregate_metrics(list(baseline_recall_eq1_ids.values()))
        compare_recall_eq1_aggregate_metrics = aggregate_metrics(list(compare_data_recall_eq1_ids.values()))

        #### union
        union = list(set(baseline_recall_eq1_ids.keys()).union(set(compare_data_recall_eq1_ids.keys())))
        union_recall_eq1_entries_in_baseline = [entry for entry in baseline_data if entry["instance_id"] in union]
        union_recall_eq1_entries_in_compare = [entry for entry in compare_data if entry["instance_id"] in union]

        recall_eq1_union_aggregate_metrics_in_baseline = aggregate_metrics(union_recall_eq1_entries_in_baseline)
        recall_eq1_union_aggregate_metrics_in_compare = aggregate_metrics(union_recall_eq1_entries_in_compare)


        #### intersection
        intersection = list(set(baseline_recall_eq1_ids.keys()).intersection(set(compare_data_recall_eq1_ids.keys())))
        intersection_recall_eq1_entries_in_baseline = [entry for entry in baseline_data if entry["instance_id"] in intersection]
        intersection_recall_eq1_entries_in_compare = [entry for entry in compare_data if entry["instance_id"] in intersection]

        recall_eq1_intersection_aggregate_metrics_in_baseline = aggregate_metrics(intersection_recall_eq1_entries_in_baseline)
        recall_eq1_intersection_aggregate_metrics_in_compare = aggregate_metrics(intersection_recall_eq1_entries_in_compare)


def main():
    """Main function to load data and generate comparisons."""
    # Load all experiments
    all_data = {}
    for name, path in LIST_OF_SUMMARY.items():
        try:
            data = load_json(path)
            all_data[name] = data
            print(f"Loaded {name}: {len(data)} entries")
        except FileNotFoundError:
            print(f"Warning: Could not find file for {name}: {path}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    if BASELINE not in all_data:
        print(f"Error: Baseline '{BASELINE}' not found in loaded data")
        return

    # Aggregate baseline stats
    baseline_stats = aggregate_metrics(all_data[BASELINE])
    print(f"\nBaseline aggregated: {baseline_stats['num_entries']} entries")
    print(json.dumps(baseline_stats, indent=4, sort_keys=False, default=str))

    # Compare each experiment to baseline
    comparisons = {}
    for name, data in all_data.items():
        if name == BASELINE:
            continue

        print("\n" + "=" * 80)
        print(f"COMPARISON: Baseline vs {name}")
        print("=" * 80)

        other_stats = aggregate_metrics(data)
        print(f"\n {name} aggregated: {other_stats['num_entries']} entries")
        print(json.dumps(other_stats, indent=4, sort_keys=False, default=str))










if __name__ == "__main__":
    main()
