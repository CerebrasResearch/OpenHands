import json
import sys

def extract_instance_ids(jsonl_file):
    """
    Extract all instance_id values from a JSONL file.

    Args:
        jsonl_file: Path to the JSONL file

    Returns:
        List of instance IDs
    """
    instance_ids = []

    try:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if 'instance_id' in data:
                        instance_ids.append(data['instance_id'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} - Invalid JSON: {e}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    return instance_ids


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ids.py <jsonl_file> [--unique] [--output <file>]")
        print("\nOptions:")
        print("  --unique    Remove duplicate instance IDs")
        print("  --output    Save results to a file (one ID per line)")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    unique = '--unique' in sys.argv

    # Extract instance IDs
    instance_ids = extract_instance_ids(jsonl_file)

    # Remove duplicates if requested
    if unique:
        instance_ids = list(dict.fromkeys(instance_ids))  # Preserves order

    # Handle output
    if '--output' in sys.argv:
        try:
            output_idx = sys.argv.index('--output')
            output_file = sys.argv[output_idx + 1]
            with open(output_file, 'w') as f:
                for instance_id in instance_ids:
                    f.write(f"{instance_id}\n")
            print(f"Extracted {len(instance_ids)} instance IDs to '{output_file}'")
        except (IndexError, ValueError):
            print("Error: --output requires a filename", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout
        for instance_id in instance_ids:
            print(instance_id)
        print(f"\nTotal: {len(instance_ids)} instance IDs", file=sys.stderr)


if __name__ == "__main__":
    main()
