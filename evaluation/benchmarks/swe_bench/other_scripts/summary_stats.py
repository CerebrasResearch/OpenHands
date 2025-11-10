import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - MUST be before importing pyplot

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
import argparse
from datetime import datetime


LIST_OF_SUMMARY = {
    "Qwen30B_Baseline_Hard50": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen30B_sb_dev_baseline_cmd_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary_selected.json",

    "Qwen30B_LocPr5_Hard50": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen30B_sb_dev_LocPr5_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary_selected.json",

    "Qwen30B_LocPr5_phase4_think_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_thinkNOstrreplace_ph4planphase_full176/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary_selected_50hard.json",

    "Qwen30B_Baseline_CepoToolv2_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_baseline_cepo_tool_v2_swebench_dev_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_Baseline_Hard50": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen480B_sb_dev_baseline_cmd_only_50inst_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-3-coder-480b_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_LocPr5_Hard50": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen480B_sb_dev_LocPr5_50inst_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-3-coder-480b_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_LocPr5_phase4_think_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_sb_dev_LocPr5_thinkNOstrreplace_ph4planphase_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_Baseline_CepoToolv2_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_baseline_cepo_tool_v2_swebench_dev_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_LocPr5_phase4_think_CepoToolv2_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_sb_dev_cepo_tool_v2_LocPr5_thinkNOstrreplace_ph4planphase_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_Baseline_CepoToolv3_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_baseline_cepo_tool_v3_swebench_dev_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_Baseline_CepoToolv4_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_baseline_cepo_tool_v4_swebench_dev_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen480B_Baseline_CepoToolv5_Hard50": "/workspaces/OpenHands/evaluation/eval_llm_qwen480B_baseline_cepo_tool_v5_swebench_dev_50hard/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct-FP8_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",


    "Qwen30B_Baseline_176": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen30B_sb_dev_baseline_cmd_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",

    "Qwen30B_LOCPr5_176": "/workspaces/OpenHands/evaluation/eval_runs_previous/eval_llm_qwen30B_sb_dev_LocPr5_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json"
}


BASELINE = "Qwen480B_Baseline_Hard50"
# Generate timestamp once at script start for consistency across all files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def load_json(json_path):
    with open(json_path, 'r') as fh:
        data = json.load(fh)
    return data


def aggregate_metrics(entries):
    """Aggregate metrics from a list of entries."""
    total_tool_calls = defaultdict(int)
    total_bash_calls = defaultdict(int)
    total_resolved = 0
    total_llm_metrics = {
        "num_calls_to_llm": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    total_localization_stats = {
        "precision": 0,
        "recall": 0
    }
    num_aggregated = {"precision": 0.0, "recall": 0.0}

    # Count perfect precision and recall
    perfect_precision_count = 0
    perfect_recall_count = 0

    precision_0p9_1p0 = 0
    recall_0p9_1p0 = 0

    num_entries = len(entries)

    for entry in entries:
        if "tool_call_counts" in entry:
            for tool, count in entry["tool_call_counts"].items():
                total_tool_calls[tool] += count

        if "bash_tool_call_counts" in entry:
            for bash_cmd, count in entry["bash_tool_call_counts"].items():
                total_bash_calls[bash_cmd] += count

        if "llm_metrics" in entry:
            for metric, value in entry["llm_metrics"].items():
                if metric in total_llm_metrics:
                    total_llm_metrics[metric] += value

        total_resolved += int(entry["is_resolved"])

        for key in ["precision", "recall"]:
            if key in entry:
                total_localization_stats[key] += entry[key]
                num_aggregated[key] += 1

        # Count perfect scores
        if entry.get("precision") == 1.0:
            perfect_precision_count += 1
        if entry.get("recall") == 1.0:
            perfect_recall_count += 1

        if entry.get("precision") >= 0.9 and entry.get("precision") <= 1.0:
            precision_0p9_1p0 += 1
        if entry.get("recall") >= 0.9 and entry.get("recall") <= 1.0:
            recall_0p9_1p0 += 1

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
        "avg_completion_tokens_per_instance": total_llm_metrics["completion_tokens"]/num_entries if num_entries else 0,
        "perfect_precision_count": perfect_precision_count,
        "perfect_recall_count": perfect_recall_count,
        "precision_0p9_1p0": precision_0p9_1p0,
        "recall_0p9_1p0": recall_0p9_1p0
    }


def filter_data(baseline_data, compare_data, filter_type="resolved"):
    """Filter data by resolved status or recall == 1.0 and compute metrics for union/intersection."""

    if filter_type == "resolved":
        baseline_ids = {entry["instance_id"]: entry for entry in baseline_data if entry["is_resolved"]}
        compare_ids = {entry["instance_id"]: entry for entry in compare_data if entry["is_resolved"]}
    elif filter_type == "recall_eq_1":
        baseline_ids = {entry["instance_id"]: entry for entry in baseline_data if entry.get("recall") == 1.0}
        compare_ids = {entry["instance_id"]: entry for entry in compare_data if entry.get("recall") == 1.0}

    baseline_individual_metrics = aggregate_metrics(list(baseline_ids.values()))
    compare_individual_metrics = aggregate_metrics(list(compare_ids.values()))

    union = list(set(baseline_ids.keys()).union(set(compare_ids.keys())))
    union_baseline_entries = [entry for entry in baseline_data if entry["instance_id"] in union]
    union_compare_entries = [entry for entry in compare_data if entry["instance_id"] in union]

    union_baseline_metrics = aggregate_metrics(union_baseline_entries)
    union_compare_metrics = aggregate_metrics(union_compare_entries)

    intersection = list(set(baseline_ids.keys()).intersection(set(compare_ids.keys())))
    intersection_baseline_entries = [entry for entry in baseline_data if entry["instance_id"] in intersection]
    intersection_compare_entries = [entry for entry in compare_data if entry["instance_id"] in intersection]

    intersection_baseline_metrics = aggregate_metrics(intersection_baseline_entries)
    intersection_compare_metrics = aggregate_metrics(intersection_compare_entries)

    return {
        "individual": {"baseline": baseline_individual_metrics, "compare": compare_individual_metrics},
        "union": {"baseline": union_baseline_metrics, "compare": union_compare_metrics, "count": len(union)},
        "intersection": {"baseline": intersection_baseline_metrics, "compare": intersection_compare_metrics, "count": len(intersection)}
    }


def generate_random_colors(num_colors):
    """Generate random distinct colors."""
    colors = []
    used_hues = set()
    max_attempts = 100  # Prevent infinite loops

    for _ in range(num_colors):
        attempts = 0
        while attempts < max_attempts:
            hue = random.uniform(0, 1)
            saturation = random.uniform(0.6, 1.0)
            value = random.uniform(0.7, 1.0)

            is_distinct = True
            for used_hue in used_hues:
                hue_diff = min(abs(hue - used_hue), 1 - abs(hue - used_hue))
                if hue_diff < 0.1:
                    is_distinct = False
                    break

            if is_distinct:
                used_hues.add(hue)
                rgb = mcolors.hsv_to_rgb([hue, saturation, value])
                hex_color = mcolors.rgb2hex(rgb)
                colors.append(hex_color)
                break

            attempts += 1

        # If we couldn't find a distinct color after max_attempts, just use a random one
        if attempts >= max_attempts:
            hue = random.uniform(0, 1)
            saturation = random.uniform(0.6, 1.0)
            value = random.uniform(0.7, 1.0)
            rgb = mcolors.hsv_to_rgb([hue, saturation, value])
            hex_color = mcolors.rgb2hex(rgb)
            colors.append(hex_color)

    return colors


def create_comparison_plots(all_stats, baseline_name="Baseline", output_dir="./"):
    """Create comparison plots for all experiments."""
    try:
        import sys
        print("\n[DEBUG] Starting create_comparison_plots")
        sys.stdout.flush()

        names = list(all_stats.keys())
        resolved_counts = [all_stats[name]['total_resolved'] for name in names]
        resolved_pcts = [100 * all_stats[name]['total_resolved'] / all_stats[name]['num_entries'] for name in names]
        avg_llm_calls = [all_stats[name]['avg_llm_calls_per_instance'] for name in names]
        avg_prompt_tokens = [all_stats[name]['avg_prompt_tokens_per_instance'] for name in names]
        avg_precision = [all_stats[name]['localization_avg_metrics']['precision'] for name in names]
        avg_recall = [all_stats[name]['localization_avg_metrics']['recall'] for name in names]
        perfect_precision = [all_stats[name]['perfect_precision_count'] for name in names]
        perfect_recall = [all_stats[name]['perfect_recall_count'] for name in names]
        precision_0p9_1p0 = [all_stats[name]['precision_0p9_1p0'] for name in names]
        recall_0p9_1p0 = [all_stats[name]['recall_0p9_1p0'] for name in names]

        # Generate random colors for each bar
        colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in names]

        print("[DEBUG] Creating first plot (Resolution metrics)")
        sys.stdout.flush()

        # First plot: Resolution metrics
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 6))
        fig1.suptitle('Experiment Comparison Report - Resolution Metrics', fontsize=16, fontweight='bold', y=0.98)

        axes1[0].bar(names, resolved_counts, color=colors, alpha=0.7, edgecolor='black')
        axes1[0].set_ylabel('Number Resolved')
        axes1[0].set_title('Issues Resolved')
        axes1[0].set_xticks(range(len(names)))
        axes1[0].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(resolved_counts):
            axes1[0].text(i, v + 0.2, str(v), ha='center', va='bottom', fontweight='bold', rotation=90)

        axes1[1].bar(names, resolved_pcts, color=colors, alpha=0.7, edgecolor='black')
        axes1[1].set_ylabel('Percentage (%)')
        axes1[1].set_title('Resolution Rate')
        axes1[1].set_xticks(range(len(names)))
        axes1[1].set_xticklabels(names, rotation=90, ha='center')
        axes1[1].set_ylim([0, max(resolved_pcts) * 1.15])
        for i, v in enumerate(resolved_pcts):
            axes1[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes1[2].bar(names, avg_llm_calls, color=colors, alpha=0.7, edgecolor='black')
        axes1[2].set_ylabel('Avg Calls')
        axes1[2].set_title('Avg LLM Calls per Instance')
        axes1[2].set_xticks(range(len(names)))
        axes1[2].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(avg_llm_calls):
            axes1[2].text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        plt.tight_layout()
        filepath1 = os.path.join(output_dir, f'comparison_report_resolution_{TIMESTAMP}.png')

        print("[DEBUG] Saving first plot")
        sys.stdout.flush()
        plt.savefig(filepath1, dpi=600, bbox_inches='tight')
        print(f"\n✓ Saved: comparison_report_resolution_{TIMESTAMP}.png")
        plt.close(fig1)

        print("[DEBUG] Creating second plot (Performance metrics)")
        sys.stdout.flush()

        # Second plot: Performance metrics
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 6))
        fig2.suptitle('Experiment Comparison Report - Performance Metrics', fontsize=16, fontweight='bold', y=0.98)

        axes2[0].bar(names, avg_prompt_tokens, color=colors, alpha=0.7, edgecolor='black')
        axes2[0].set_ylabel('Avg Tokens')
        axes2[0].set_title('Avg Prompt Tokens per Instance')
        axes2[0].set_xticks(range(len(names)))
        axes2[0].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(avg_prompt_tokens):
            axes2[0].text(i, v + 60, f'{v:.0f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes2[1].bar(names, avg_precision, color=colors, alpha=0.7, edgecolor='black')
        axes2[1].set_ylabel('Precision')
        axes2[1].set_title('Avg Precision')
        axes2[1].set_ylim([0, 1.1])
        axes2[1].set_xticks(range(len(names)))
        axes2[1].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(avg_precision):
            axes2[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes2[2].bar(names, avg_recall, color=colors, alpha=0.7, edgecolor='black')
        axes2[2].set_ylabel('Recall')
        axes2[2].set_title('Avg Recall')
        axes2[2].set_ylim([0, 1.1])
        axes2[2].set_xticks(range(len(names)))
        axes2[2].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(avg_recall):
            axes2[2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        plt.tight_layout()
        filepath2 = os.path.join(output_dir, f'comparison_report_performance_{TIMESTAMP}.png')

        print("[DEBUG] Saving second plot")
        sys.stdout.flush()
        plt.savefig(filepath2, dpi=600, bbox_inches='tight')
        print(f"\n✓ Saved: comparison_report_performance_{TIMESTAMP}.png")
        plt.close(fig2)

        print("[DEBUG] Creating third plot (Perfect Precision and Recall)")
        sys.stdout.flush()

        # Third plot: Perfect Precision and Recall
        fig3, axes3 = plt.subplots(2, 2, sharex=True, figsize=(15, 15))
        fig3.suptitle('Experiment Comparison Report - Precision and Recall Counts', fontsize=16, fontweight='bold', y=0.98)

        axes3[0][0].bar(names, perfect_precision, color=colors, alpha=0.7, edgecolor='black')
        axes3[0][0].set_ylabel('Count')
        axes3[0][0].set_title('Precision==1')
        axes3[0][0].set_ylim([0, 500])
        axes3[0][0].set_yticks(np.arange(0, 501, 100))
        axes3[0][0].set_xticks(range(len(names)))
        axes3[0][0].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(perfect_precision):
            axes3[0][0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes3[0][1].bar(names, perfect_recall, color=colors, alpha=0.7, edgecolor='black')
        axes3[0][1].set_ylabel('Count')
        axes3[0][1].set_title('Recall==1.0')
        axes3[0][1].set_ylim([0, 500])
        axes3[0][1].set_yticks(np.arange(0, 501, 100))
        axes3[0][1].set_xticks(range(len(names)))
        axes3[0][1].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(perfect_recall):
            axes3[0][1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes3[1][0].bar(names, precision_0p9_1p0, color=colors, alpha=0.7, edgecolor='black')
        axes3[1][0].set_ylabel('Count')
        axes3[1][0].set_title('0.9<=Precision<=1')
        axes3[1][0].set_ylim([0, 500])
        axes3[1][0].set_yticks(np.arange(0, 501, 100))
        axes3[1][0].set_xticks(range(len(names)))
        axes3[1][0].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(precision_0p9_1p0):
            axes3[1][0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        axes3[1][1].bar(names, recall_0p9_1p0, color=colors, alpha=0.7, edgecolor='black')
        axes3[1][1].set_ylabel('Count')
        axes3[1][1].set_title('0.9<=Recall<=1.0')
        axes3[1][1].set_ylim([0, 500])
        axes3[1][1].set_yticks(np.arange(0, 501, 100))
        axes3[1][1].set_xticks(range(len(names)))
        axes3[1][1].set_xticklabels(names, rotation=90, ha='center')
        for i, v in enumerate(recall_0p9_1p0):
            axes3[1][1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', rotation=90)

        plt.tight_layout()
        filepath3 = os.path.join(output_dir, f'comparison_report_perfect_metrics_{TIMESTAMP}.png')

        print("[DEBUG] Saving third plot")
        sys.stdout.flush()
        plt.savefig(filepath3, dpi=600, bbox_inches='tight')
        print(f"\n✓ Saved: comparison_report_perfect_metrics_{TIMESTAMP}.png")
        plt.close(fig3)

        print("[DEBUG] Completed create_comparison_plots")
        sys.stdout.flush()

    except Exception as e:
        print(f"✗ Error in create_comparison_plots: {e}")
        import traceback
        traceback.print_exc()


def create_grand_total_tool_calls_plot(all_stats, baseline_name="Baseline", output_dir="./"):
    """Create plot for grand total tool calls."""
    try:
        import sys
        print("[DEBUG] Inside create_grand_total_tool_calls_plot")
        sys.stdout.flush()

        names = list(all_stats.keys())
        grand_totals = [all_stats[name]['grand_total_tool_calls'] for name in names]
        colors_list = generate_random_colors(len(names))

        print("[DEBUG] Creating figure")
        sys.stdout.flush()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, grand_totals, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)

        print("[DEBUG] Adding text labels")
        sys.stdout.flush()

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12, rotation=90)

        ax.set_ylabel('Total Tool Calls', fontsize=12, fontweight='bold')
        ax.set_title('Grand Total Tool Calls per Experiment', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(axis='y', alpha=0.3)

        print("[DEBUG] Tight layout and saving")
        sys.stdout.flush()

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'grand_total_tool_calls_{TIMESTAMP}.png')

        print(f"[DEBUG] Saving to {filepath}")
        sys.stdout.flush()

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: grand_total_tool_calls_{TIMESTAMP}.png")

        print("[DEBUG] Closing figure")
        sys.stdout.flush()

        plt.close(fig)

        print("[DEBUG] Exiting create_grand_total_tool_calls_plot")
        sys.stdout.flush()

    except Exception as e:
        print(f"✗ Error in create_grand_total_tool_calls_plot: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()


def create_tool_calls_all_tools_plot(all_stats, baseline_name="Baseline", output_dir="./"):
    """Create plot for all tools called."""
    try:
        names = list(all_stats.keys())

        all_tools = set()
        for name in names:
            all_tools.update(all_stats[name]['total_tool_calls'].keys())
        all_tools = sorted(list(all_tools))

        colors_list = generate_random_colors(len(names))

        # Create one subplot per tool
        n_cols = 3
        n_rows = (len(all_tools) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()

        for tool_idx, tool in enumerate(all_tools):
            ax = axes[tool_idx]
            y = np.arange(len(names))

            tool_counts = [all_stats[name]['total_tool_calls'].get(tool, 0) for name in names]
            bars = ax.barh(y, tool_counts, 0.6, color=colors_list, alpha=0.8, edgecolor='black')

            for bar_idx, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    ax.text(width + max(tool_counts) * 0.01, bar.get_y() + bar.get_height()/2., f'{int(width)}',
                           ha='left', va='center', fontsize=9, fontweight='bold')

            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.set_xlabel('Total Calls', fontsize=10, fontweight='bold')
            ax.set_title(f'Tool: {tool}', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        # Hide extra subplots
        for idx in range(len(all_tools), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Tool Calls by Experiment', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'tool_calls_all_tools_comparison_{TIMESTAMP}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: tool_calls_all_tools_comparison_{TIMESTAMP}.png")
        plt.close(fig)

    except Exception as e:
        print(f"✗ Error in create_tool_calls_all_tools_plot: {e}")
        import traceback
        traceback.print_exc()


def create_tool_calls_per_tool_plots(all_stats, baseline_name="Baseline", output_dir="./"):
    """Create individual plots for each tool."""
    try:
        names = list(all_stats.keys())
        colors_list = generate_random_colors(len(names))

        all_tools = set()
        for name in names:
            all_tools.update(all_stats[name]['total_tool_calls'].keys())
        all_tools = sorted(list(all_tools))

        for tool in all_tools:
            fig, ax = plt.subplots(figsize=(10, 6))

            tool_counts = [all_stats[name]['total_tool_calls'].get(tool, 0) for name in names]
            bars = ax.bar(names, tool_counts, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                           ha='center', va='bottom', fontweight='bold', fontsize=12)

            ax.set_ylabel('Total Calls', fontsize=12, fontweight='bold')
            ax.set_title(f'Tool: {tool} - Calls Across Experiments', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', alpha=0.3)

            safe_tool_name = tool.replace('/', '_').replace(' ', '_')
            filename = f'tool_{safe_tool_name}_{TIMESTAMP}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"✓ Saved {len(all_tools)} individual tool plots")

    except Exception as e:
        print(f"✗ Error in create_tool_calls_per_tool_plots: {e}")
        import traceback
        traceback.print_exc()


def create_filter_analysis_plots(all_data, all_stats, baseline_name="Baseline", output_dir="./"):
    """Create comparison plots for filter analysis."""
    try:
        names = [name for name in all_stats.keys() if name != baseline_name]
        colors_list = generate_random_colors(len(names) + 1)
        color_map = {baseline_name: colors_list[0]}
        for i, name in enumerate(names):
            color_map[name] = colors_list[i + 1]

        baseline_data = all_data[baseline_name]

        for compare_name, compare_data in all_data.items():
            if compare_name == baseline_name:
                continue

            for filter_type in ["resolved", "recall_eq_1"]:
                filter_results = filter_data(baseline_data, compare_data, filter_type)
                filter_label = "Resolved" if filter_type == "resolved" else "Recall=1.0"

                exp_names = [baseline_name, compare_name]
                exp_colors = [color_map[baseline_name], color_map[compare_name]]

                individual_baseline = filter_results["individual"]["baseline"]
                individual_compare = filter_results["individual"]["compare"]
                union_count = filter_results["union"]["count"]
                union_baseline = filter_results["union"]["baseline"]
                union_compare = filter_results["union"]["compare"]
                intersection_count = filter_results["intersection"]["count"]
                intersection_baseline = filter_results["intersection"]["baseline"]
                intersection_compare = filter_results["intersection"]["compare"]

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'{filter_label} Analysis: {baseline_name} vs {compare_name}', fontsize=16, fontweight='bold')

                individual_llm_calls = [individual_baseline['avg_llm_calls_per_instance'],
                                        individual_compare['avg_llm_calls_per_instance']]
                union_llm_calls = [union_baseline['avg_llm_calls_per_instance'],
                                  union_compare['avg_llm_calls_per_instance']]
                intersection_llm_calls = [intersection_baseline['avg_llm_calls_per_instance'],
                                         intersection_compare['avg_llm_calls_per_instance']]

                x_pos = np.arange(len(exp_names))
                width = 0.35

                for ax, llm_calls, title_suffix, count in [(axes[0], individual_llm_calls,
                                                             f'Individual\n(Count: {individual_baseline["num_entries"]} vs {individual_compare["num_entries"]})', None),
                                                            (axes[1], union_llm_calls, f'Union\n(Total: {union_count})', None),
                                                            (axes[2], intersection_llm_calls, f'Intersection\n(Common: {intersection_count})', None)]:
                    bars = ax.bar(x_pos - width/2, llm_calls, width, color=exp_colors, alpha=0.8, edgecolor='black')
                    ax.set_ylabel('Avg LLM Calls', fontsize=11, fontweight='bold')
                    ax.set_title(f'{filter_label} {title_suffix}', fontsize=12, fontweight='bold')
                    ax.set_xticks(x_pos - width/2)
                    ax.set_xticklabels(exp_names, rotation=90)
                    ax.grid(axis='y', alpha=0.3)

                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')

                plt.tight_layout()
                safe_compare_name = compare_name.replace(' ', '_').replace('/', '_')
                filename = f'filter_{filter_type}_{baseline_name}_{safe_compare_name}_{TIMESTAMP}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f"✓ Saved filter analysis plots")

    except Exception as e:
        print(f"✗ Error in create_filter_analysis_plots: {e}")
        import traceback
        traceback.print_exc()


def print_metrics_report(name, stats):
    """Print formatted metrics report."""
    print(f"\n{'='*80}")
    print(f"REPORT: {name}")
    print(f"{'='*80}")
    print(f"Total Entries: {stats['num_entries']}")
    print(f"Resolved: {stats['total_resolved']} / {stats['num_entries']} ({100*stats['total_resolved']/stats['num_entries']:.1f}%)")
    print(f"\nLLM Metrics:")
    print(f"  Avg LLM Calls per Instance: {stats['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens per Instance: {stats['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens per Instance: {stats['avg_completion_tokens_per_instance']:.2f}")
    print(f"\nLocalization Metrics:")
    print(f"  Avg Precision: {stats['localization_avg_metrics']['precision']:.4f}")
    print(f"  Avg Recall: {stats['localization_avg_metrics']['recall']:.4f}")
    print(f"\nTool Calls:")
    print(f"  Grand Total Tool Calls: {stats['grand_total_tool_calls']}")
    print(f"  Top 5 Tools:")
    sorted_tools = sorted(stats['total_tool_calls'].items(), key=lambda x: x[1], reverse=True)[:5]
    for tool, count in sorted_tools:
        print(f"    {tool}: {count}")


def print_comparison_report(baseline_stats, compare_stats, baseline_name, compare_name):
    """Print formatted comparison report."""
    print(f"\n{'='*80}")
    print(f"COMPARISON ACROSS CATEGORIES: {baseline_name} vs {compare_name}")
    print(f"{'='*80}")

    resolved_improvement = compare_stats['total_resolved'] - baseline_stats['total_resolved']
    resolved_pct_improvement = (resolved_improvement / baseline_stats['total_resolved'] * 100) if baseline_stats['total_resolved'] > 0 else 0

    llm_calls_ratio = compare_stats['avg_llm_calls_per_instance'] / baseline_stats['avg_llm_calls_per_instance'] if baseline_stats['avg_llm_calls_per_instance'] > 0 else 0
    prompt_tokens_ratio = compare_stats['avg_prompt_tokens_per_instance'] / baseline_stats['avg_prompt_tokens_per_instance'] if baseline_stats['avg_prompt_tokens_per_instance'] > 0 else 0
    completion_tokens_ratio = compare_stats['avg_completion_tokens_per_instance'] / baseline_stats['avg_completion_tokens_per_instance'] if baseline_stats['avg_completion_tokens_per_instance'] > 0 else 0

    print(f"\nResolved Issues:")
    print(f"  Baseline: {baseline_stats['total_resolved']} / {baseline_stats['num_entries']}")
    print(f"  {compare_name}: {compare_stats['total_resolved']} / {compare_stats['num_entries']}")
    print(f"  Difference: {resolved_improvement:+d} ({resolved_pct_improvement:+.1f}%)")

    print(f"\nLLM Efficiency:")
    print(f"  LLM Calls per Instance: {baseline_stats['avg_llm_calls_per_instance']:.2f} → {compare_stats['avg_llm_calls_per_instance']:.2f} ({llm_calls_ratio:.2f}x)")
    print(f"  Prompt Tokens per Instance: {baseline_stats['avg_prompt_tokens_per_instance']:.2f} → {compare_stats['avg_prompt_tokens_per_instance']:.2f} ({prompt_tokens_ratio:.2f}x)")
    print(f"  Completion Tokens per Instance: {baseline_stats['avg_completion_tokens_per_instance']:.2f} → {compare_stats['avg_completion_tokens_per_instance']:.2f} ({completion_tokens_ratio:.2f}x)")

    print(f"\nLocalization Quality:")
    precision_diff = compare_stats['localization_avg_metrics']['precision'] - baseline_stats['localization_avg_metrics']['precision']
    recall_diff = compare_stats['localization_avg_metrics']['recall'] - baseline_stats['localization_avg_metrics']['recall']
    print(f"  Precision: {baseline_stats['localization_avg_metrics']['precision']:.4f} → {compare_stats['localization_avg_metrics']['precision']:.4f} ({precision_diff:+.4f})")
    print(f"  Recall: {baseline_stats['localization_avg_metrics']['recall']:.4f} → {compare_stats['localization_avg_metrics']['recall']:.4f} ({recall_diff:+.4f})")


def print_filter_analysis(baseline_data, compare_data, baseline_name, compare_name, filter_type="resolved"):
    """Print detailed analysis of filtered data (union/intersection)."""
    import sys

    print(f"\n[DEBUG] Starting filter analysis: {baseline_name} vs {compare_name}, type: {filter_type}")
    sys.stdout.flush()

    filter_results = filter_data(baseline_data, compare_data, filter_type)

    print(f"[DEBUG] Filter data computed")
    sys.stdout.flush()

    filter_label = "Resolved Issues" if filter_type == "resolved" else "Perfect Recall (recall=1.0)"

    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {filter_label}")
    print(f"Comparing {baseline_name} vs {compare_name}")
    print(f"{'='*80}")
    sys.stdout.flush()

    # Individual Statistics
    print(f"\n--- INDIVIDUAL {filter_label.upper()} ---")
    sys.stdout.flush()
    baseline_ind = filter_results["individual"]["baseline"]
    compare_ind = filter_results["individual"]["compare"]

    print(f"\n{baseline_name}:")
    print(f"  Count: {baseline_ind['num_entries']}")
    print(f"  Avg LLM Calls: {baseline_ind['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {baseline_ind['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {baseline_ind['avg_completion_tokens_per_instance']:.2f}")
    print(f"  Avg Precision: {baseline_ind['localization_avg_metrics']['precision']:.4f}")
    print(f"  Avg Recall: {baseline_ind['localization_avg_metrics']['recall']:.4f}")
    sys.stdout.flush()

    print(f"\n{compare_name}:")
    print(f"  Count: {compare_ind['num_entries']}")
    print(f"  Avg LLM Calls: {compare_ind['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {compare_ind['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {compare_ind['avg_completion_tokens_per_instance']:.2f}")
    print(f"  Avg Precision: {compare_ind['localization_avg_metrics']['precision']:.4f}")
    print(f"  Avg Recall: {compare_ind['localization_avg_metrics']['recall']:.4f}")
    sys.stdout.flush()

    print(f"[DEBUG] Finished individual stats")
    sys.stdout.flush()

    # Union Analysis
    union_count = filter_results["union"]["count"]
    print(f"\n--- UNION ANALYSIS (Total {filter_label.lower()}: {union_count}) ---")
    union_baseline = filter_results["union"]["baseline"]
    union_compare = filter_results["union"]["compare"]

    print(f"\n{baseline_name} on union set:")
    print(f"  Avg LLM Calls: {union_baseline['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {union_baseline['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {union_baseline['avg_completion_tokens_per_instance']:.2f}")

    print(f"\n{compare_name} on union set:")
    print(f"  Avg LLM Calls: {union_compare['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {union_compare['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {union_compare['avg_completion_tokens_per_instance']:.2f}")

    llm_calls_ratio = union_compare['avg_llm_calls_per_instance'] / union_baseline['avg_llm_calls_per_instance'] if union_baseline['avg_llm_calls_per_instance'] > 0 else 0
    tokens_ratio = union_compare['avg_prompt_tokens_per_instance'] / union_baseline['avg_prompt_tokens_per_instance'] if union_baseline['avg_prompt_tokens_per_instance'] > 0 else 0

    print(f"\n  LLM Calls Ratio: {llm_calls_ratio:.2f}x")
    print(f"  Prompt Tokens Ratio: {tokens_ratio:.2f}x")

    # Intersection Analysis
    intersection_count = filter_results["intersection"]["count"]
    print(f"\n--- INTERSECTION ANALYSIS (Common {filter_label.lower()}: {intersection_count}) ---")
    intersection_baseline = filter_results["intersection"]["baseline"]
    intersection_compare = filter_results["intersection"]["compare"]

    print(f"\n{baseline_name} on intersection set:")
    print(f"  Avg LLM Calls: {intersection_baseline['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {intersection_baseline['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {intersection_baseline['avg_completion_tokens_per_instance']:.2f}")

    print(f"\n{compare_name} on intersection set:")
    print(f"  Avg LLM Calls: {intersection_compare['avg_llm_calls_per_instance']:.2f}")
    print(f"  Avg Prompt Tokens: {intersection_compare['avg_prompt_tokens_per_instance']:.2f}")
    print(f"  Avg Completion Tokens: {intersection_compare['avg_completion_tokens_per_instance']:.2f}")

    if intersection_baseline['avg_llm_calls_per_instance'] > 0:
        llm_calls_ratio_intersect = intersection_compare['avg_llm_calls_per_instance'] / intersection_baseline['avg_llm_calls_per_instance']
        tokens_ratio_intersect = intersection_compare['avg_prompt_tokens_per_instance'] / intersection_baseline['avg_prompt_tokens_per_instance']
        print(f"\n  LLM Calls Ratio: {llm_calls_ratio_intersect:.2f}x")
        print(f"  Prompt Tokens Ratio: {tokens_ratio_intersect:.2f}x")


def main():
    """Main function to load data and generate comprehensive reports."""
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment comparison reports')
    parser.add_argument('-o', '--output', type=str, default='./comparison_output',
                       help='Output folder to save plots and reports (default: ./comparison_output)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {TIMESTAMP}")
    print("Loading experiment data...")

    all_data = {}
    for name, path in LIST_OF_SUMMARY.items():
        try:
            data = load_json(path)
            all_data[name] = data
            print(f"  ✓ Loaded {name}: {len(data)} entries")
        except FileNotFoundError:
            print(f"  ✗ Warning: Could not find file for {name}")
        except Exception as e:
            print(f"  ✗ Error loading {name}: {e}")

    if BASELINE not in all_data:
        print(f"✗ Error: Baseline not found")
        return

    # Aggregate stats for all experiments
    all_stats = {}
    for name, data in all_data.items():
        all_stats[name] = aggregate_metrics(data)

    # Print individual reports
    for name, stats in all_stats.items():
        print_metrics_report(name, stats)

    # Print comparison reports
    baseline_stats = all_stats[BASELINE]
    for name, stats in all_stats.items():
        if name != BASELINE:
            print_comparison_report(baseline_stats, stats, BASELINE, name)

    # Print detailed filter analysis
    baseline_data = all_data[BASELINE]
    print("\n[DEBUG] Starting filter analysis loop")
    import sys
    sys.stdout.flush()

    for name, compare_data in all_data.items():
        if name != BASELINE:
            print(f"\n[DEBUG] Processing {name}")
            sys.stdout.flush()
            print_filter_analysis(baseline_data, compare_data, BASELINE, name, filter_type="resolved")
            print(f"[DEBUG] Completed resolved filter for {name}")
            sys.stdout.flush()
            print_filter_analysis(baseline_data, compare_data, BASELINE, name, filter_type="recall_eq_1")
            print(f"[DEBUG] Completed recall_eq_1 filter for {name}")
            sys.stdout.flush()

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    import sys
    sys.stdout.flush()

    print("[DEBUG] Calling create_comparison_plots")
    sys.stdout.flush()
    create_comparison_plots(all_stats, BASELINE, output_dir)

    print("[DEBUG] Calling create_grand_total_tool_calls_plot")
    sys.stdout.flush()
    create_grand_total_tool_calls_plot(all_stats, BASELINE, output_dir)

    print("[DEBUG] Calling create_tool_calls_all_tools_plot")
    sys.stdout.flush()
    create_tool_calls_all_tools_plot(all_stats, BASELINE, output_dir)

    print("[DEBUG] Calling create_tool_calls_per_tool_plots")
    sys.stdout.flush()
    create_tool_calls_per_tool_plots(all_stats, BASELINE, output_dir)

    print("[DEBUG] Calling create_filter_analysis_plots")
    sys.stdout.flush()
    create_filter_analysis_plots(all_data, all_stats, BASELINE, output_dir)

    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")
    print(f"Timestamp used: {TIMESTAMP}")


if __name__ == "__main__":
    main()
