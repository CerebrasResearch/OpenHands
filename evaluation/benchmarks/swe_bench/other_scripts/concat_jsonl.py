#!/usr/bin/env python3
"""
Script to concatenate multiple JSONL files, filtering out lines with retry errors.
Lines containing "error": "Maximum retries exceeded" are moved to a separate error file.
"""

import sys
import json
import argparse

def process_and_concatenate_jsonl(input_files, output, error_output):
    """
    Concatenate multiple JSONL files, moving error lines to a separate file.
    Lines with "error": "Maximum retries exceeded" are separated out.
    """
    error_count = 0
    success_count = 0

    try:
        with open(output, 'w') as outfile, open(error_output, 'w') as errfile:
            # Process each input file in order
            for input_file in input_files:
                with open(input_file, 'r') as infile:
                    for line in infile:
                        if should_move_to_errors(line):
                            errfile.write(line)
                            error_count += 1
                        else:
                            outfile.write(line)
                            success_count += 1

        print(f"Successfully processed files:")
        print(f"  - {success_count} lines written to {output}")
        print(f"  - {error_count} error lines written to {error_output}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def should_move_to_errors(line):
    """Check if a line contains the retry error."""
    try:
        data = json.loads(line.strip())
        error = data.get('error')
        return error is not None and "Maximum retries" in error
    except json.JSONDecodeError:
        # If line is not valid JSON, keep it in main file
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Concatenate multiple JSONL files, filtering out retry errors to a separate file.'
    )
    parser.add_argument(
        '--input_files',
        nargs='+',
        help='Input JSONL files to concatenate'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file for concatenated results'
    )
    parser.add_argument(
        '-e', '--errors',
        dest='error_output',
        default='errors.jsonl',
        help='Output file for error lines (default: errors.jsonl)'
    )

    args = parser.parse_args()

    print(f"Input files: {args.input_files}")
    print(f"Output file: {args.output}")
    print(f"Error file: {args.error_output}")
    process_and_concatenate_jsonl(
        args.input_files,
        args.output,
        args.error_output
    )

if __name__ == "__main__":
    main()
