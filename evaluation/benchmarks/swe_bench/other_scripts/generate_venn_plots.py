import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations
import numpy as np
import json


LIST_OF_SUMMARY = {
    "Baseline_50": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_baseline_cmd_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary_selected.json",
    "LOCPr5_Qwen30B_50": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary_selected.json",
    "Baseline_176": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_baseline_cmd_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",
    "LOCPr5_Qwen30B_176": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_all/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",
    "LOCPr5_str_replace_edit_think_plan_brainstorm_Qwen30B": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_str_replace_think_plan_50inst/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json",
    "LocPr5_str_replace_think_plan_resolvedbaselocpr": "/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_str_replace_think_plan_resolvedbaselocpr/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/overall_summary.json"
}

BASELINE = "Baseline_50"

def load_json(json_path):
    with open(json_path, 'r') as fh:
        data = json.load(fh)
    return data


def create_venn_diagram(data, title="Venn Diagram", output_file=None):
    """
    Create a Venn diagram for multiple sets.

    Parameters:
    -----------
    data : dict
        Dictionary where keys are set names and values are sets of elements.
        Example: {'Set A': {1, 2, 3}, 'Set B': {2, 3, 4}, 'Set C': {3, 4, 5}}

    title : str
        Title for the diagram

    output_file : str, optional
        Path to save the figure. If None, displays the plot.

    Returns:
    --------
    dict : Dictionary containing counts for each intersection

    Example:
    --------
    data = {
        'Agentless': {1, 2, 3, 4, 5, 6, 7, 8},
        'Open-source': {3, 4, 5, 6, 8, 9, 10, 11, 12},
        'Commercial': {2, 6, 7, 8, 13, 14}
    }
    create_venn_diagram(data, "Issue Fixes Comparison")
    """

    n_sets = len(data)
    set_names = list(data.keys())
    sets = [data[name] for name in set_names]

    # Validate input
    if n_sets < 2:
        raise ValueError("At least 2 sets are required for a Venn diagram")
    if n_sets > 6:
        raise ValueError("This function supports up to 6 sets. For more, consider using specialized libraries like matplotlib-venn or upsetplot")

    # Calculate all possible intersections
    intersection_counts = {}

    for i in range(1, 2**n_sets):
        # Generate binary representation to determine which sets to intersect
        subset_indices = [j for j in range(n_sets) if (i >> j) & 1]

        # Calculate intersection
        if subset_indices:
            intersection = sets[subset_indices[0]].copy()
            for idx in subset_indices[1:]:
                intersection &= sets[idx]

            # Create key as frozenset for hashability
            key = frozenset(subset_indices)
            intersection_counts[key] = len(intersection)

    # Create visualization based on number of sets
    if n_sets == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        _plot_2_set_venn(ax, set_names, intersection_counts)
    elif n_sets == 3:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        _plot_3_set_venn(ax, set_names, intersection_counts)
    elif n_sets >= 4:
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        _plot_n_set_venn(ax, set_names, intersection_counts, n_sets)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Diagram saved to {output_file}")
    else:
        plt.show()

    return intersection_counts



def _plot_n_set_venn(ax, names, counts, n_sets):
    """Plot N-set Venn diagram with circles arranged in a circle"""
    # Generate circle positions arranged in a circle
    angle_step = 2 * np.pi / n_sets
    radius = 0.3
    center_x, center_y = 0.5, 0.5

    colors = plt.cm.Set3(np.linspace(0, 1, n_sets))
    circles = []

    for i in range(n_sets):
        angle = i * angle_step
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        circles.append({'center': (x, y), 'color': colors[i], 'label': names[i], 'index': i})

        circle = patches.Circle((x, y), 0.15, fill=True, alpha=0.2,
                               color=colors[i], ec=colors[i], linewidth=2)
        ax.add_patch(circle)

    # Add text labels for set names
    for i, circle_info in enumerate(circles):
        x, y = circle_info['center']
        # Place labels outside circles
        label_dist = 0.25
        label_x = center_x + label_dist * np.cos(i * angle_step)
        label_y = center_y + label_dist * np.sin(i * angle_step)
        ax.text(label_x, label_y, circle_info['label'], fontsize=11, ha='center',
                va='center', fontweight='bold', color=circle_info['color'])

    # Add intersection counts - place them throughout the space
    for key, count in counts.items():
        set_indices = sorted(list(key))

        # Calculate position as average of involved circles
        if len(set_indices) == 1:
            # Single set - place at circle perimeter
            idx = set_indices[0]
            angle = idx * angle_step
            x = center_x + (radius + 0.12) * np.cos(angle)
            y = center_y + (radius + 0.12) * np.sin(angle)
        else:
            # Multiple sets - place at centroid
            x = sum(circles[idx]['center'][0] for idx in set_indices) / len(set_indices)
            y = sum(circles[idx]['center'][1] for idx in set_indices) / len(set_indices)

        ax.text(x, y, str(count), fontsize=11, ha='center', va='center',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')


def _plot_2_set_venn(ax, names, counts):
    """Plot 2-set Venn diagram"""
    circle1 = patches.Circle((0.3, 0.5), 0.2, fill=True, alpha=0.3, color='blue', ec='blue', linewidth=2)
    circle2 = patches.Circle((0.7, 0.5), 0.2, fill=True, alpha=0.3, color='red', ec='red', linewidth=2)

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Add text labels
    ax.text(0.2, 0.5, str(counts.get(frozenset([0]), 0)), fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.5, str(counts.get(frozenset([0, 1]), 0)), fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(0.8, 0.5, str(counts.get(frozenset([1]), 0)), fontsize=14, ha='center', va='center', fontweight='bold')

    ax.text(0.2, 0.75, names[0], fontsize=12, ha='center', fontweight='bold')
    ax.text(0.8, 0.75, names[1], fontsize=12, ha='center', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def _plot_3_set_venn(ax, names, counts):
    """Plot 3-set Venn diagram"""
    # Circle positions for 3-set Venn
    circles = [
        {'center': (0.25, 0.5), 'color': 'blue', 'label': names[0]},
        {'center': (0.75, 0.5), 'color': 'red', 'label': names[1]},
        {'center': (0.5, 0.25), 'color': 'green', 'label': names[2]}
    ]

    for circle in circles:
        c = patches.Circle(circle['center'], 0.2, fill=True, alpha=0.2,
                          color=circle['color'], ec=circle['color'], linewidth=2)
        ax.add_patch(c)

    # Text positions for 3-set (adjusted for readability)
    text_positions = {
        frozenset([0]): (0.15, 0.65),           # Only A
        frozenset([1]): (0.85, 0.65),           # Only B
        frozenset([2]): (0.5, 0.05),            # Only C
        frozenset([0, 1]): (0.5, 0.6),          # A ∩ B
        frozenset([0, 2]): (0.35, 0.35),        # A ∩ C
        frozenset([1, 2]): (0.65, 0.35),        # B ∩ C
        frozenset([0, 1, 2]): (0.5, 0.4)        # A ∩ B ∩ C
    }

    for key, pos in text_positions.items():
        count = counts.get(key, 0)
        ax.text(pos[0], pos[1], str(count), fontsize=12, ha='center', va='center', fontweight='bold')

    # Add set labels
    ax.text(0.1, 0.8, names[0], fontsize=12, ha='center', fontweight='bold', color='blue')
    ax.text(0.9, 0.8, names[1], fontsize=12, ha='center', fontweight='bold', color='red')
    ax.text(0.5, 0.0, names[2], fontsize=12, ha='center', fontweight='bold', color='green')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def _plot_4_set_venn(ax, names, counts):
    """Plot 4-set Venn diagram (rectangular overlap)"""
    # Simplified 4-set representation
    rect1 = patches.Rectangle((0.1, 0.2), 0.35, 0.6, fill=True, alpha=0.2,
                              color='blue', ec='blue', linewidth=2, label=names[0])
    rect2 = patches.Rectangle((0.55, 0.2), 0.35, 0.6, fill=True, alpha=0.2,
                              color='red', ec='red', linewidth=2, label=names[1])
    circle1 = patches.Circle((0.25, 0.65), 0.15, fill=True, alpha=0.2,
                            color='green', ec='green', linewidth=2, label=names[2])
    circle2 = patches.Circle((0.75, 0.65), 0.15, fill=True, alpha=0.2,
                            color='orange', ec='orange', linewidth=2, label=names[3])

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)


def main():
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

    resolved_ids = dict()
    for key, entries in all_data.items():
        resolved = [entry["instance_id"] for entry in entries if entry["is_resolved"]]
        resolved_ids[key] = set(resolved)

    create_venn_diagram(resolved_ids, "resolved_overlap", output_file="venn.png")





# Example usage
if __name__ == "__main__":
    main()
    # # Example 1: 3-set Venn diagram
    # data_3 = {
    #     'Agentless': {1, 2, 3, 4, 5, 6, 7, 8},
    #     'Open-source': {3, 4, 5, 6, 8, 9, 10, 11, 12},
    #     'Commercial': {2, 6, 7, 8, 13, 14}
    # }

    # counts = create_venn_diagram(data_3, "Issue Fixes Comparison - 3 Sets")

    # print("\nIntersection Counts:")
    # set_names = list(data_3.keys())
    # for key in sorted(counts.keys()):
    #     set_indices = sorted(key)
    #     involved_sets = [set_names[i] for i in set_indices]
    #     print(f"  {' ∩ '.join(involved_sets)}: {counts[key]}")

    # # Example 2: 2-set Venn diagram
    # print("\n" + "="*50 + "\n")
    # data_2 = {
    #     'Python': {1, 2, 3, 4, 5},
    #     'JavaScript': {4, 5, 6, 7, 8}
    # }

    # create_venn_diagram(data_2, "Programming Languages Overlap")
