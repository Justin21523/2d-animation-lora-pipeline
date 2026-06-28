#!/usr/bin/env bash
#
# Test Runner for 2D Animation LoRA Pipeline
#
# Runs the CPU-safe demo/smoke test suite
#
# Author: LLMProvider Tooling
# Date: 2025-01-17

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "2D Animation LoRA Pipeline Smoke Test Suite"
echo "========================================"
echo ""

# Get project root (parent of tests directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Count tests
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test file
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file")

    echo -e "${YELLOW}Running: $test_name${NC}"

    if python -m pytest "$test_file" -v --tb=short; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED: $test_name${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Run CPU-safe smoke tests
echo "=== CPU-safe smoke tests ==="
echo ""

if [ -f "tests/demo/test_demo_manifest.py" ]; then
    run_test "tests/demo/test_demo_manifest.py"
fi

if [ -f "tests/test_config.py" ]; then
    run_test "tests/test_config.py"
fi

if [ -f "tests/test_end_to_end_pipeline.py" ]; then
    run_test "tests/test_end_to_end_pipeline.py"
fi

if [ -f "tests/core/pipeline/test_resource_monitor.py" ]; then
    run_test "tests/core/pipeline/test_resource_monitor.py"
fi

if [ -f "tests/core/pipeline/test_stage_manager.py" ]; then
    run_test "tests/core/pipeline/test_stage_manager.py"
fi

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total:  $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"

if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    exit 1
else
    echo "Failed: 0"
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
