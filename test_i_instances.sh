#!/bin/bash
mkdir -p outputs

sa_iterations=10000

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--parallel)
            PARALLEL=1
            shift
            ;;
        --iterations)
            sa_iterations="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ "$PARALLEL" == "1" ]]; then
    echo "Mode: Parallel"
    make par
else
    echo "Mode: Sequential"
    make
fi

echo "SA iterations: $sa_iterations"

result_file="outputs/result_i_instances.csv"

angles=(30)

for infile in I_testcases/*.txt; do
    [ -e "$infile" ] || continue

    filename=$(basename "$infile")
    best_outfile="outputs/${filename}.out"

    best_angle=""
    best_cost=999999999.0
    best_err_output=""

    echo "Processing $infile..."

    for angle in "${angles[@]}"; do
        temp_outfile="outputs/${filename}_${angle}.tmp"
        temp_errfile="outputs/${filename}_${angle}.err"

        ./solve_cvrptw "$infile" "$angle" 1000 "$sa_iterations" > "$temp_outfile" 2> "$temp_errfile"

        final_cost=$(grep -oP "Final_Cost:\s+\K[0-9.]+" "$temp_errfile")
        total_time=$(grep -oP "Total_Time:\s+\K[0-9.]+" "$temp_errfile")

        if [ -z "$final_cost" ]; then
            echo "  [Angle $angle] Failed to extract a valid cost."
            rm -f "$temp_outfile" "$temp_errfile"
            continue
        fi

        echo "  [Angle $angle] Cost = $final_cost | Time = $total_time seconds"

        is_better=$(echo "$final_cost < $best_cost" | bc -l)

        if [ "$is_better" -eq 1 ]; then
            best_cost=$final_cost
            best_angle=$angle

            mv "$temp_outfile" "$best_outfile"
            best_err_output=$(cat "$temp_errfile")
        else
            rm -f "$temp_outfile"
        fi

        rm -f "$temp_errfile"
    done

    if [ -n "$best_angle" ]; then
        echo "-> Best result for $filename is Angle: $best_angle (Cost: $best_cost)"
        echo "$best_err_output" >> "$result_file"
    else
        echo "-> Could not find a valid solution for $filename."
    fi
    echo "---------------------------------------------------"
done

echo "All files processed. Best results are in outputs/ and $result_file"
