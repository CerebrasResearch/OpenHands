import os
import json
import glob
import re

def load_json(file):
    with open(file, "r") as fh:
        data = json.load(fh)
    return data

def get_all_subfolder_paths(directory_path):
    subfolder_paths = []
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for dirname in dirnames:
            subfolder_paths.append(os.path.join(dirpath, dirname))
    return subfolder_paths


def list_all_matching_files(base_path, pattern=None):
    """
    List all files matching the pattern with their numeric values.

    Args:
        base_path: Root directory to search from
        pattern: Optional regex pattern

    Returns:
        File path with maximum numeric value (or None if no matches)
    """
    if pattern is None:
        pattern = r'openai__.*?-(\d+(?:\.\d+)?).json$'

    matching_files = []

    # Walk through directory tree
    for root, dirs, files in os.walk(base_path):
        for file in files:
            match = re.search(pattern, file)
            if match:
                numeric_value = float(match.group(1))
                full_path = os.path.join(root, file)
                matching_files.append((full_path, numeric_value))

    # Sort by numeric value
    matching_files.sort(key=lambda x: x[1], reverse=True)

    # Return only the path (not the tuple)
    return matching_files[0][0] if matching_files else None


def zip_by_dirname(list1, list2, keep_unmatched=False):
    """
    Zip two lists of paths based on matching directory names.

    Args:
        list1: First list of file paths
        list2: Second list of file paths
        keep_unmatched: If True, include unmatched entries with None as pair

    Returns:
        List of tuples (path_from_list1, path_from_list2) for matching dirnames
    """
    # Create dictionary mapping dirname to path for list2
    dirname_to_path2 = {os.path.dirname(path): path for path in list2 if path is not None}

    # Create zipped list
    zipped = []
    matched_dirnames = set()

    for path1 in list1:
        dirname = os.path.dirname(path1)
        if dirname in dirname_to_path2:
            zipped.append((path1, dirname_to_path2[dirname]))
            matched_dirnames.add(dirname)
        elif keep_unmatched:
            zipped.append((path1, None))

    # Add unmatched from list2 if requested
    if keep_unmatched:
        for path2 in list2:
            if path2 is not None:
                dirname = os.path.dirname(path2)
                if dirname not in matched_dirnames:
                    zipped.append((None, path2))

    return zipped


def get_files(folder):
    trajectories = glob.glob(os.path.join(folder, "*/*trajectory.json"))

    subfolders = get_all_subfolder_paths(folder)
    llm_paths = []

    for sub in subfolders:
        llm_path = list_all_matching_files(sub)
        if llm_path:  # Only add if a match was found
            llm_paths.append(llm_path)

    print(f"LEN of trajectories: {len(trajectories)}")
    print(f"LEN of llm_paths: {len(llm_paths)}")

    zipped_traj_llm_paths = zip_by_dirname(trajectories, llm_paths, keep_unmatched=False)

    print(f"LEN of matched pairs: {len(zipped_traj_llm_paths)}\n")

    for traj_path, llm_path in zipped_traj_llm_paths:
        print(f"path trajectory = {traj_path}")
        print(f"path llm_path = {llm_path}")

        traj = load_json(traj_path)
        num_actions = 0
        num_obs = 0
        for item in traj:
            action = item.get('action', '')
            observation = item.get('observation', '')
            if action and item.get('source') == 'agent':
                num_actions +=1
            elif observation and item.get('source') == 'agent':
                num_obs += 1

        llm_msg = load_json(llm_path)

        len_traj = len(traj)
        num_messages = len(llm_msg["messages"])

        print(f"len_trajectory_action_obs = {len_traj}, len_llm: {num_messages}, diff={len_traj-num_messages}\n")
        print(f"num_actions: {num_actions}, num_obs = {num_obs}, diff = {num_actions - num_obs}")
        print(f"--------------------------\n")


if __name__ == "__main__":
    get_files("/workspaces/OpenHands/evaluation/eval_llm_qwen30B_sb_dev_LocPr5_str_replace_think_plan_50inst/outputs/princeton-nlp__SWE-bench-dev/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.56.0-no-hint-run_1/llm_completions")
