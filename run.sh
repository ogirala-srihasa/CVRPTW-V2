#!/bin/bash
mkdir -p outputs

sa_iterations=10000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations)
            sa_iterations="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

make

echo "SA iterations: $sa_iterations"

result_file="outputs/result.csv"
angle=30

for infile in testcase/*; do
    filename=$(basename "$infile")
    outfile="outputs/${filename}.out"
    ./solve_cvrptw "$infile" "$angle" "$sa_iterations" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done
echo "All files processed. Results are in outputs/result.csv"
