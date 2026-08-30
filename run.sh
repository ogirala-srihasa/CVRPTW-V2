#!/bin/bash
mkdir -p outputs
make

result_file="outputs/result.csv"
angle=30

for infile in testcase/*; do
    filename=$(basename "$infile")
    outfile="outputs/${filename}.out"
    ./solve_cvrptw "$infile" "$angle" > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done
echo "All files processed. Results are in outputs/result.csv"