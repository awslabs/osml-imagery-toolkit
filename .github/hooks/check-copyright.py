#!/usr/bin/env python3

"""Check for the presence of copyright notice in files."""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import yaml


def check_copyright(content: str, copyright_pattern: re.Pattern, header_lines: int = 5) -> bool:
    """Check if the file header contains a copyright notice.

    Args:
        content: The file content to check
        copyright_pattern: Compiled regex pattern to match
        header_lines: Number of lines to check at the start of file

    Returns:
        bool: True if copyright notice is found in the header
    """
    # Get the first n lines of the file
    header = "\n".join(content.splitlines()[:header_lines])
    return bool(copyright_pattern.search(header))


def main() -> None:
    """Check copyright notices in files."""
    parser = argparse.ArgumentParser(description="Check for copyright notices in files")
    parser.add_argument("--config", help="Path to config file", default=".github/copyright.yaml")
    parser.add_argument("files", nargs="+", help="Files to check")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Compile copyright pattern
    copyright_pattern = re.compile(config.get("copyright_pattern"), re.IGNORECASE)
    # Get number of header lines to check from config, default to 5
    header_lines = config.get("header_lines", 5)
    # Get failure mode from config, default to True (fail on missing copyright)
    fail_on_missing = config.get("fail_on_missing", True)

    # Check each file
    failed_files: List[str] = []
    for file_path in args.files:
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                if not check_copyright(content, copyright_pattern, header_lines):
                    failed_files.append(file_path)
                    print(f"No copyright notice found in first {header_lines} lines of {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            failed_files.append(file_path)

    if failed_files:
        print("\nFiles missing copyright notice:")
        for file in failed_files:
            print(f"  {file}")

        if fail_on_missing:
            print("\nCopyright check failed - missing copyright notices")
            sys.exit(1)
        else:
            print("\nWarning: Files are missing copyright notices but check is configured to warn only")
            sys.exit(0)

    print("Copyright check passed for all files")


if __name__ == "__main__":
    main()
