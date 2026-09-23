#!/bin/bash
# run_cuopt.sh
#
# Runs the cuOpt GPU solver on every instance in a testcase folder, giving
# each instance the same wall-clock budget our own solver spent on it. Those
# budgets come from the Final_Time column of our results CSV, so our solver
# must have been run over the same instances first.
#
# Results are written to outputs_cuopt/cuopt_results.csv
#
# Usage:
#   bash run_cuopt.sh                                             # testcase/ vs outputs/result.csv
#   bash run_cuopt.sh I_testcases outputs/result_i_instances.csv  # other instance set

set -euo pipefail

TESTCASE_DIR="${1:-testcase}"
RESULTS_CSV="${2:-outputs/result.csv}"
OUTPUT_DIR="outputs_cuopt"
OUTPUT_CSV="${OUTPUT_DIR}/cuopt_results.csv"

if [ ! -f "${RESULTS_CSV}" ]; then
    echo "ERROR: ${RESULTS_CSV} not found." >&2
    echo "Run our solver over these instances first (e.g. bash test.sh --parallel)" >&2
    echo "so the per-instance time limits exist, then re-run this script." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running cuOpt over instances in ${TESTCASE_DIR}/"
echo "Per-instance time limits taken from ${RESULTS_CSV} (Final_Time column)"

python3 solve_cuopt.py \
    --testcase_dir "${TESTCASE_DIR}" \
    --results_csv "${RESULTS_CSV}" \
    --output_csv "${OUTPUT_CSV}"

echo "All instances processed. Results are in ${OUTPUT_CSV}"
