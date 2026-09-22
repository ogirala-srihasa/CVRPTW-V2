#!/bin/bash
mkdir -p outputs

cluster_size=1000
sa_rm_iterations=10000
sa_only_iterations=10000
post_opt_iterations=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster-size)
            cluster_size="$2"
            shift 2
            ;;
        --sa-rm-iterations)
            sa_rm_iterations="$2"
            shift 2
            ;;
        --sa-iterations)
            sa_only_iterations="$2"
            shift 2
            ;;
        --post-opt-iterations)
            post_opt_iterations="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

make

echo "Cluster size: $cluster_size"
echo "SA+RM iterations: $sa_rm_iterations"
echo "SA-only iterations: $sa_only_iterations"
echo "Post-opt iterations: $post_opt_iterations"

result_file="outputs/result.csv"
csv_header="File,Construction_Time,Construction_Cost,Construction_Vehicles,Merge_Time,Merge_Cost,Merge_Vehicles,PostOpt_Time,PostOpt_Cost,PostOpt_Vehicles,Final_Time,Final_Cost,Final_Vehicles,Valid"

if [ ! -f "$result_file" ]; then
    echo "$csv_header" > "$result_file"
fi

for infile in testcase/*; do
    [ -e "$infile" ] || continue

    filename=$(basename "$infile")
    outfile="outputs/${filename}.out"

    ./solve_cvrptw "$infile" "$cluster_size" "$sa_rm_iterations" \
        "$sa_only_iterations" "$post_opt_iterations" \
        > "$outfile" 2>> "$result_file"
    echo "Processed $infile -> $outfile"
done

echo "All files processed. Results are in $result_file"
