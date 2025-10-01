#!/usr/bin/env python3
"""
auto_edit_distance_analysis.py

Automatically computes functional edit distances, per-file distances, and turn summaries,
categorizes instances, and saves results in organized folders for a given model.
Also generates per-category and combined histograms (grouped, KDE, boxplots) for num_iters, start_id, and end_id.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -----------------------
# Cleaning & distance
# -----------------------
def clean_patch(patch: str) -> str:
    if not patch:
        return ""
    lines = []
    for line in patch.splitlines():
        if line.startswith("@@"):
            m = re.search(r'@@.*@@\s+(.*)', line)
            if m:
                signature = m.group(1).strip()
                if signature.startswith(("class ", "def ")):
                    lines.append(signature)
            continue
        if line.startswith(("diff --git", "index", "---", "+++")):
            continue
        code_line = line[1:] if line and line[0] in "+- " else line
        lines.append(code_line)
    code = "\n".join(lines)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)

    def normalise_line(ln: str) -> str:
        if not ln.strip():
            return ""
        indent = len(ln) - len(ln.lstrip(" "))
        content = " ".join(ln.lstrip().split())
        return " " * indent + content

    cleaned_lines = [normalise_line(ln) for ln in code.splitlines() if ln.strip()]
    return "\n".join(cleaned_lines)


def compute_functional_edit_distance(a: str, b: str) -> Tuple[int, float]:
    a_clean = clean_patch(a)
    b_clean = clean_patch(b)
    return Levenshtein.distance(a_clean, b_clean), SequenceMatcher(None, a_clean, b_clean).ratio()


# -----------------------
# Patch handling
# -----------------------
def split_patch_by_file(patch: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not patch:
        return files
    file_blocks = re.findall(r'(^diff --git a/.*?)(?=^diff --git a/|\Z)', patch, flags=re.MULTILINE | re.DOTALL)
    for block in file_blocks:
        m = re.search(r'^diff --git a/(.*?) b/', block, flags=re.MULTILINE)
        filename = m.group(1).strip() if m else f"__FULL_PATCH__#{len(files)+1}"
        files[filename] = block.strip("\n")
    return files


def compute_per_file_comparison(gold_patch: str, agent_patch: str) -> Tuple[Dict[str, Any], Any, Any]:
    gold_files = split_patch_by_file(gold_patch)
    agent_files = split_patch_by_file(agent_patch)
    all_files = set(gold_files) | set(agent_files)
    results: Dict[str, Any] = {}
    matched_distances: List[int] = []
    matched_similarities: List[float] = []

    for fn in sorted(all_files):
        gold = gold_files.get(fn)
        agent = agent_files.get(fn)
        if gold is not None and agent is not None:
            dist, sm_ratio = compute_functional_edit_distance(gold, agent)
            results[fn] = {"status": "matched", "edit_distance": dist, "similarity_ratio": sm_ratio}
            matched_distances.append(dist)
            matched_similarities.append(sm_ratio)
        elif gold is not None:
            results[fn] = {"status": "missing_in_generated_patch", "edit_distance": None, "similarity_ratio": None}
        else:
            results[fn] = {"status": "extra_in_generated_patch", "edit_distance": None, "similarity_ratio": None}

    avg_dist = sum(matched_distances) / len(matched_distances) if matched_distances else None
    avg_ratio = sum(matched_similarities) / len(matched_similarities) if matched_similarities else None
    return results, avg_dist, avg_ratio


# -----------------------
# JSON extraction helpers
# -----------------------
def extract_problematic_ids(data: Dict[str, Any]) -> set:
    ids = set()
    stats = data.get("swe_bench_statistics") or data
    for k in ("unresolved_ids", "error_ids"):
        v = stats.get(k)
        if isinstance(v, list):
            ids.update(map(str, v))
    return ids


def extract_recall_one_ids(data: Any) -> Tuple[set, set]:
    recall_1, recall_lt1 = set(), set()
    items = data.items() if isinstance(data, dict) else [(str(i.get("instance_id") or i.get("id")), i) for i in data if isinstance(i, dict)]
    for k, v in items:
        r = float(v.get("recall", 0))
        (recall_1 if r == 1.0 else recall_lt1).add(str(k))
    return recall_1, recall_lt1


def extract_patch_mapping(data: Any) -> Dict[str, Dict[str, Any]]:
    mapping = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            inst = item["instance"].get("instance_id")
            mapping[str(inst)] = {
                "patch": item["instance"].get("patch"),
                "generated_patch": item["test_result"].get("git_patch"),
                "history": item.get("history", [])
            }
    return mapping


def get_instance_turn_summary(history: List[Dict[str, Any]], per_file_summary: Dict[str, Any]) -> Dict[str, Any]:
    files_matched = [f.split('/')[-1] for f, v in per_file_summary.items() if v['status'] == 'matched']
    turns_summary = {
        'total_turns': -2,
        'history_length': len(history),
        'files_changed': files_matched,
        'files_summary': {file: {'start_id': None, 'end_id': None, 'num_iters_required': 0, 'turns': []} for file in files_matched}
    }
    for item in history:
        action = item.get('action','')
        if not action:
            continue
        turns_summary['total_turns'] += 1
        for file in files_matched:
            content = ''
            if action == 'run' and file in item.get('message', '') and f'test_{file}' not in item.get('message', ''):
                content = item['message']
            elif action == 'edit':
                content = item.get('tool_call_metadata', {}).get('model_response', {}).get('choices', [{}])[0]\
                    .get('message', {}).get('tool_calls', [{}])[0].get('function', {}).get('arguments', '')
                if file not in content or f'test_{file}' in content:
                    continue
            if content:
                turn_info = turns_summary['files_summary'][file]
                if turn_info['start_id'] is None:
                    turn_info['start_id'] = item['id']
                turn_info['end_id'] = item['id']
                turn_info['num_iters_required'] += 1
                turn_info['turns'].append({'id': item['id'], 'action': action, 'message': content})
    return turns_summary


# -----------------------
# Path resolution
# -----------------------
def resolve_model_paths(base_dir: Path, model_name: str) -> Tuple[Path, Path, Path]:
    # base = base_dir / model_name / "evaluations"
    # swe_dir = base / "swe_bench_evals"
    # problematic_ids_json = list(swe_dir.glob("*.json"))[0]
    # loc_dir = base / "localisation"
    # recall_jsonl = list(loc_dir.glob("*.jsonl"))[0]
    # patch_data_json = base / "bash_tool_evals" / "json" / "Non-empty_patches_filtered.json"

    problematic_ids_json = list(base_dir.glob(f"final_eval/consolidated_report_llm.qwen_coder_30b_small*.json"))[0]
    recall_jsonl = list(base_dir.glob(f"localization/*localisation_report.jsonl"))[0]
    patch_data_json = list(base_dir.glob(f"bash_tool_call_summary_llm.qwen_coder_30b_small/json/Non-empty_patches_filtered.json"))[0]

    # problematic_ids_json = Path("/mlf11-shared/coding/aarti_oh_2/OpenHands/evaluation/eval_base_tempdefault_cmdTr_locFa/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_250_N_v0.56.0-no-hint-run_1/final_eval/consolidated_report_llm.qwen_coder_30b_small_20250926_213340.json")

    # recall_jsonl = Path("/mlf11-shared/coding/aarti_oh_2/OpenHands/evaluation/eval_base_tempdefault_cmdTr_locFa/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_250_N_v0.56.0-no-hint-run_1/localization/llm.qwen_coder_30b_small_localisation_report.jsonl")

    # patch_data_json = Path("/mlf11-shared/coding/aarti_oh_2/OpenHands/evaluation/eval_base_tempdefault_cmdTr_locFa/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_250_N_v0.56.0-no-hint-run_1/bash_tool_call_summary_llm.qwen_coder_30b_small/json/Non-empty_patches_filtered.json")

    return problematic_ids_json, recall_jsonl, patch_data_json


# -----------------------
# Main driver
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Automatic functional edit distance + turn summary analysis.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--base_dir", default=".")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    model_name = args.model_name
    output_base = base_dir / model_name / "evaluations" / "distance_and_turn_evals"
    output_base.mkdir(exist_ok=True, parents=True)

    problematic_ids_json, recall_jsonl, patch_data_json = resolve_model_paths(base_dir, model_name)
    data1 = json.loads(problematic_ids_json.read_text(encoding="utf-8"))
    data2 = [json.loads(line) for line in recall_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    data3 = json.loads(patch_data_json.read_text(encoding="utf-8"))

    problematic_ids = extract_problematic_ids(data1)
    recall_1_ids, recall_lt1_ids = extract_recall_one_ids(data2)
    patch_map = extract_patch_mapping(data3)

    categories = {
        "unresolved_with_100_percent_recall": problematic_ids & recall_1_ids,
        "resolved_with_100_percent_recall": recall_1_ids - problematic_ids,
        "unresolved_with_less_than_100_recall": problematic_ids & recall_lt1_ids,
        "resolved_with_less_than_100_recall": recall_lt1_ids - problematic_ids
    }

    combined_stats = {"num_iters_required": {}, "start_id": {}, "end_id": {}}

    for cat_name, inst_ids in categories.items():
        cat_dir = output_base / cat_name
        cat_dir.mkdir(exist_ok=True, parents=True)
        summary, turn_file, file_distance = {}, {}, {}
        per_instance_avgs, per_instance_ratios = [], []

        for inst_id in sorted(inst_ids):
            inst_data = patch_map.get(inst_id)
            if not inst_data:
                continue
            per_file, avg_dist, avg_ratio = compute_per_file_comparison(inst_data["patch"], inst_data["generated_patch"])
            summary[inst_id] = {"per_file": per_file, "average_edit_distance": avg_dist, "average_similarity_ratio": avg_ratio}
            turn_file[inst_id] = get_instance_turn_summary(inst_data["history"], per_file)
            file_distance[inst_id] = per_file
            if avg_dist is not None: per_instance_avgs.append(avg_dist)
            if avg_ratio is not None: per_instance_ratios.append(avg_ratio)

        (cat_dir / "turn_file.json").write_text(json.dumps(turn_file, indent=2))
        all_num_iters, all_start_ids, all_end_ids = [], [], []
        for inst, files_data in turn_file.items():
            for file, details in files_data.get("files_summary", {}).items():
                if details.get("num_iters_required") is not None: all_num_iters.append(details["num_iters_required"])
                if details.get("start_id") is not None: all_start_ids.append(details["start_id"])
                if details.get("end_id") is not None: all_end_ids.append(details["end_id"])

        def save_hist(data, title, xlabel, filename, bins=20):
            if not data: return
            plt.figure(figsize=(8, 5))
            plt.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency"); plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout(); plt.savefig(cat_dir / filename, dpi=300); plt.close()

        save_hist(all_num_iters, f"Distribution of num_iters_required for {model_name}", "num_iters_required", "num_iters_hist.png")
        save_hist(all_start_ids, f"Distribution of start_id for {model_name}", "start_id", "start_id_hist.png")
        save_hist(all_end_ids, f"Distribution of end_id for {model_name}", "end_id", "end_id_hist.png")

        combined_stats["num_iters_required"][cat_name] = all_num_iters
        combined_stats["start_id"][cat_name] = all_start_ids
        combined_stats["end_id"][cat_name] = all_end_ids

    # -----------------------
    # Combined analysis plots
    # -----------------------
    combined_dir = output_base / "combined_histograms"
    combined_dir.mkdir(exist_ok=True, parents=True)

    def plot_grouped_hist(data_dict, title, xlabel, filename, bins=20):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        all_data = [v for v in data_dict.values() if v]
        if not all_data: return
        all_concat = sum(all_data, [])
        counts, bin_edges, _ = plt.hist(all_concat, bins=bins)  # <-- fixed unpack
        plt.clf()
        width = (bin_edges[1] - bin_edges[0]) / (len(data_dict) + 1)
        for i, (cat, data) in enumerate(data_dict.items()):
            counts, _, _ = plt.hist(data, bins=bin_edges, alpha=0)  # only get counts, hide plot
            plt.bar(bin_edges[:-1] + i * width, counts, width=width, color=colors[i], label=f"{cat} (n={len(data)})")
        plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency"); plt.legend(fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(combined_dir / filename, dpi=300); plt.close()

    def plot_kde(data_dict, title, xlabel, filename):
        plt.figure(figsize=(10, 6))
        for i, (cat, data) in enumerate(data_dict.items()):
            if len(data) > 1:
                sns.kdeplot(data, label=f"{cat} (n={len(data)})", fill=False, linewidth=2)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel("Density"); plt.legend(fontsize=8); plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(combined_dir / filename, dpi=300); plt.close()

    def plot_boxplot(data_dict, title, xlabel, filename):
        df = pd.DataFrame([{"Category": cat, "Value": v} for cat, vals in data_dict.items() for v in vals])
        if df.empty: return
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Category", y="Value", data=df, palette="Set2")
        plt.title(title); plt.ylabel(xlabel); plt.xticks(rotation=15); plt.tight_layout()
        plt.savefig(combined_dir / filename, dpi=300); plt.close()

    for metric in ["num_iters_required", "start_id", "end_id"]:
        plot_grouped_hist(combined_stats[metric], f"Grouped Histogram {model_name}- {metric}", metric, f"{metric}_grouped.png")
        plot_kde(combined_stats[metric], f"KDE Distribution {model_name}- {metric}", metric, f"{metric}_kde.png")
        plot_boxplot(combined_stats[metric], f"Boxplot {model_name}- {metric}", metric, f"{metric}_boxplot.png")

    print(f"📊 All combined plots saved in: {combined_dir}")


if __name__ == "__main__":
    main()
