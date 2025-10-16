import os
import glob
import toml
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Find incomplete instances by comparing trajectory files with selected IDs"
    )

    parser.add_argument(
        "--llm_folders",
        type=str,
        nargs='+',
        required=True,
        help="Paths to the LLM completions folders containing trajectory files (space-separated)"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config TOML file containing selected_ids"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output file where incomplete IDs will be written (default: incomplete.txt in current directory)"
    )

    args = parser.parse_args()

    # Build output file path if not provided
    if args.output_file is None:
        args.output_file = "incomplete.txt"

    # Find all trajectory files across all LLM folders
    complete_instance = set()
    for llm_folder in args.llm_folders:
        trajs = glob.glob(os.path.join(llm_folder, "*/*trajectory.json"))
        instances = [os.path.basename(os.path.dirname(x)) for x in trajs]
        complete_instance.update(instances)
        print(f"Found {len(instances)} complete instances in {llm_folder}")

    # Load selected IDs from config
    selected_ids = toml.load(args.config_file)["selected_ids"]

    # Find incomplete IDs
    incomplete_ids = [x for x in selected_ids if x not in complete_instance]

    # Write to output file
    with open(args.output_file, "w") as fh:
        fh.write("\n".join(incomplete_ids))

    print(f"\nTotal complete instances across all folders: {len(complete_instance)}")
    print(f"Found {len(incomplete_ids)} incomplete instances")
    print(f"Incomplete IDs written to: {args.output_file}")


if __name__ == "__main__":
    main()
