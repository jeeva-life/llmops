#!/usr/bin/env python3
"""
Test runner script for Document Portal project
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run tests based on the specified type
    
    Args:
        test_type (str): Type of tests to run ('all', 'unit', 'integration', 'fast')
        verbose (bool): Run with verbose output
        coverage (bool): Run with coverage reporting
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov=api", "--cov-report=html", "--cov-report=term"])
    
    # Add test type filters
    if test_type == "unit":
        cmd.append("-m unit")
    elif test_type == "integration":
        cmd.append("-m integration")
    elif test_type == "fast":
        cmd.append("-m 'not slow'")
    elif test_type == "api":
        cmd.append("-m api")
    elif test_type == "llm":
        cmd.append("-m llm")
    elif test_type == "faiss":
        cmd.append("-m faiss")
    elif test_type != "all":
        print(f"Unknown test type: {test_type}")
        print("Available types: all, unit, integration, fast, api, llm, faiss")
        return False
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print(f"✅ All {test_type} tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"❌ Some {test_type} tests failed!")
        return False


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run Document Portal tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast", "api", "llm", "faiss"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run with verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from project root.")
        sys.exit(1)
    
    # Run tests
    success = run_tests(args.type, args.verbose, args.coverage)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
