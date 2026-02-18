"""
Test runner script for integration and E2E tests.

This script provides convenient commands to run different test suites:
- Integration tests
- End-to-end tests
- Performance tests
- Load tests
- Security tests
- All tests

Usage:
    python run_integration_tests.py [test_type]
    
Test types:
    integration - Run integration tests only
    e2e - Run end-to-end tests only
    performance - Run performance tests only
    load - Run load tests only
    security - Run security tests only
    all - Run all integration and E2E tests (default)
    quick - Run quick smoke tests
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py::TestFullWorkflow",
        "-v", "-s"
    ]
    return run_command(cmd, "Integration Tests")


def run_e2e_tests():
    """Run end-to-end tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py",
        "-v", "-s"
    ]
    return run_command(cmd, "End-to-End Tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py::TestPerformance",
        "-v", "-s"
    ]
    return run_command(cmd, "Performance Tests")


def run_load_tests():
    """Run load tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_load_performance.py",
        "-v", "-s"
    ]
    return run_command(cmd, "Load Tests")


def run_security_tests():
    """Run security tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py::TestSecurity",
        "-v", "-s"
    ]
    return run_command(cmd, "Security Tests")


def run_all_tests():
    """Run all integration and E2E tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py",
        "tests/test_load_performance.py",
        "-v", "-s"
    ]
    return run_command(cmd, "All Integration and E2E Tests")


def run_quick_tests():
    """Run quick smoke tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py::TestFullWorkflow::test_complete_submission_workflow",
        "tests/test_integration_e2e.py::TestSecurity::test_unauthorized_access_prevention",
        "-v", "-s"
    ]
    return run_command(cmd, "Quick Smoke Tests")


def run_with_coverage():
    """Run all tests with coverage report."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration_e2e.py",
        "tests/test_load_performance.py",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ]
    return run_command(cmd, "All Tests with Coverage")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integration and E2E tests for Project Stylos"
    )
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["integration", "e2e", "performance", "load", "security", "all", "quick", "coverage"],
        help="Type of tests to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Project Stylos - Integration Test Runner                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Map test types to functions
    test_runners = {
        "integration": run_integration_tests,
        "e2e": run_e2e_tests,
        "performance": run_performance_tests,
        "load": run_load_tests,
        "security": run_security_tests,
        "all": run_all_tests,
        "quick": run_quick_tests,
        "coverage": run_with_coverage
    }
    
    # Run selected tests
    runner = test_runners[args.test_type]
    success = runner()
    
    # Print summary
    print(f"\n{'='*80}")
    if success:
        print("✅ Test suite completed successfully!")
    else:
        print("❌ Test suite failed. Please check the output above.")
    print(f"{'='*80}\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
