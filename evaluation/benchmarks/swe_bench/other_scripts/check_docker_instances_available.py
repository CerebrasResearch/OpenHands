import subprocess
import json
from datasets import load_dataset
from typing import List, Dict
import argparse
import logging
from datetime import datetime

def setup_logging(log_file: str) -> logging.Logger:
    """
    Setup logging to both file and console.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger('swebench_checker')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def check_docker_image_exists(instance_id: str, use_cli: bool = True, verbose: bool = False, logger: logging.Logger = None) -> bool:
    """
    Check if a Docker image exists on Docker Hub for the given instance_id.

    Args:
        instance_id: The SWE-bench instance ID (e.g., 'astropy__astropy-11693')
        use_cli: If True, use docker CLI; if False, use Docker Hub API
        verbose: If True, print the full image name being checked

    Returns:
        True if the image exists with 'latest' tag, False otherwise
    """

    docker_image_prefix = 'docker.io/swebench/'
    repo, name = instance_id.split('__')
    tag = 'latest'
    docker_image_name = f'sweb.eval.x86_64.{repo}_1776_{name}'.lower()

    if verbose and logger:
        logger.info(f"  Checking: {docker_image_name}")

    if use_cli:
        try:
            # Use Docker Registry HTTP API v2 - works without authentication for public repos
            import requests

            # Docker Hub Registry API v2 endpoint
            url = f"https://hub.docker.com/v2/repositories/swebench/{docker_image_name}/tags/{tag}"

            if verbose and logger:
                logger.info(f"  API URL: {url}")

            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            if verbose and logger:
                logger.error(f"  Error checking {instance_id}: {e}")
            return False
    else:
        # Fallback to API method
        import requests
        url = f"https://hub.docker.com/v2/repositories/swebench/{docker_image_name}/tags/{tag}"
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except requests.RequestException as e:
            if verbose and logger:
                logger.error(f"  Error checking {instance_id}: {e}")
            return False

def get_available_instances(dataset_name: str, split: str, use_cli: bool = True,
                           limit: int = None, delay: float = 0.0, verbose: bool = False,
                           logger: logging.Logger = None) -> List[Dict]:
    """
    Get all instance IDs from specified dataset that have available Docker images.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        use_cli: If True, use docker CLI; if False, use Docker Hub API
        limit: Optional limit on number of instances to check (for testing)
        delay: Delay between checks (only needed for API method)
        verbose: If True, print full image names being checked
        logger: Logger instance for output

    Returns:
        List of dictionaries containing instance information
    """
    msg = f"Loading dataset '{dataset_name}' (split: {split})..."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    dataset = load_dataset(dataset_name, split=split)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    available_instances = []
    total = len(dataset)

    method = "Docker Hub API"
    msg = f"Checking {total} instances using {method}..."
    if logger:
        logger.info(msg)
        logger.info("-" * 60)
    else:
        print(msg)
        print("-" * 60)

    for idx, example in enumerate(dataset, 1):
        instance_id = example['instance_id']

        msg = f"[{idx}/{total}] Checking {instance_id}..."
        if logger:
            logger.info(msg)
        else:
            print(msg, end=" ", flush=True)

        if check_docker_image_exists(instance_id, use_cli=use_cli, verbose=verbose, logger=logger):
            result_msg = "✓ Available"
            if logger:
                logger.info(f"  {result_msg}")
            else:
                print(result_msg)
            available_instances.append({
                'instance_id': instance_id,
                'repo': example.get('repo', 'N/A'),
                'problem_statement': example.get('problem_statement', '')[:100] + '...'
            })
        else:
            result_msg = "✗ Not found"
            if logger:
                logger.info(f"  {result_msg}")
            else:
                print(result_msg)

        # Rate limiting
        if delay > 0 and idx < total:
            import time
            time.sleep(delay)

    return available_instances

def main():
    """Main function to run the checker."""
    parser = argparse.ArgumentParser(
        description='Check Docker image availability for SWE-bench instances using Docker CLI or API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check test split (uses Docker Hub API - no authentication needed)
  python script.py --dataset princeton-nlp/SWE-bench --split test

  # Check first 10 instances for testing
  python script.py --dataset princeton-nlp/SWE-bench --split test --limit 10

  # Check dev split with verbose output
  python script.py --dataset princeton-nlp/SWE-bench --split dev --verbose

Note: Uses Docker Hub Registry API (no Docker CLI or authentication required).
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='HuggingFace dataset name (default: princeton-nlp/SWE-bench)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to use (default: test)'
    )

    parser.add_argument(
        '--no-cli',
        action='store_true',
        help='(Deprecated - API is now default) Use Docker Hub API'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of instances to check (useful for testing)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay in seconds between API calls (default: 0.5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='available_instances.txt',
        help='Output file name (default: available_instances.txt)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default='swebench_checker.log',
        help='Log file name (default: swebench_checker.log)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print full Docker image names being checked'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    # Check dependencies
    try:
        import requests
    except ImportError:
        logger.error("Error: requests library required. Install with: pip install requests")
        exit(1)

    logger.info("SWE-bench Docker Image Availability Checker")
    logger.info("=" * 60)
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"Output file: {args.output}")
    logger.info("")

    available = get_available_instances(
        dataset_name=args.dataset,
        split=args.split,
        use_cli=True,  # Always use API now (use_cli variable is misleading but kept for compatibility)
        limit=args.limit,
        delay=args.delay,
        verbose=args.verbose,
        logger=logger
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"\nSummary: {len(available)} instances have available Docker images\n")

    if available:
        logger.info("Available Instance IDs:")
        logger.info("-" * 60)
        for item in available:
            logger.info(f"  • {item['instance_id']} ({item['repo']})")

        # Save to file
        logger.info(f"\nSaving results to '{args.output}'...")
        with open(args.output, 'w') as f:
            for item in available:
                f.write(f"{item['instance_id']}\n")
        logger.info("Done!")
    else:
        logger.info("No instances with available Docker images found.")

if __name__ == "__main__":
    main()
