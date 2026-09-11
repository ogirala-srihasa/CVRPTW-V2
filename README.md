# CVRPTW Solver

A C++ solver for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW), parallelized with OpenMP. Targets the [Gehring & Homberger 1000-customer benchmark instances](https://www.sintef.no/projectweb/top/vrptw/1000-customers/).

## Solver Pipeline

1. **Read instance** from a Solomon-format file (`lib/vrp.cpp`).
2. **Cluster customers** using angle-sweep clustering (`lib/cluster/clustering.cpp`). The sweep partitions customers by polar angle from the depot, grouping them into clusters bounded by angular range. Clusters are also hard-capped at 500 customers to prevent O(n²) blowup in downstream C&W construction. A parallel variant (`clustering_angle_sweep_parallel`) evaluates many random starting angles via OpenMP.
3. **Construct initial routes** within each cluster using the Clarke-Wright savings heuristic (`lib/clark/clarke_wright.cpp`), respecting both capacity and time-window constraints. Sequential and parallel variants available. Routes are bookended with DEPOT nodes after construction.
4. **Inter-route optimization** (`lib/optim/inter_route_optimization.cpp`). Applies three between-route improvement operators in sequence: relocate (move a customer from one route to another), swap (exchange customers between routes), and 2-opt* (reconnect route tails). Sequential and OpenMP-parallel variants.
5. **Intra-route optimization** (`lib/optim/intra_route_optimization.cpp`). Applies within-route improvement: nearest-neighbor TSP approximation followed by 2-opt. DEPOT bookends are stripped before this phase (since `postProcessIt` reorders all route elements) and re-added afterward. Sequential and OpenMP-parallel variants.
6. **Merged SA + Route Minimization** (`lib/optim/sa_optimization.cpp`). Combines vehicle-count reduction with SA-based cost improvement in a single loop (default 1,000 iterations). Each iteration:
   1. Sort routes by capacity utilization ascending (lowest-utilized first).
   2. **Early exit**: if `routes[0]` (least utilized) is at 100% utilization, all routes are fully packed — break.
   3. **Bounds check**: if walking index `i+1 >= num_routes`, reset `i = 0`.
   4. **Utilization check**: if route `i` or `i+1` is at 100% utilization, skip ejection (don't dismantle well-packed routes), run only SA operators, reset `i = 0` for next iteration.
   5. Otherwise: eject routes `i` and `i+1`, attempt cheapest feasible reinsertion of their customers into remaining routes (parallel search, tightest time-window customers first). Leftovers consolidated via sequential Clarke-Wright (`clarke_wright_cvrptw`). Advance `i += 2`.
   6. Apply SA operators: best inter-route relocate, swap, 2-opt*, and intra-route 2-opt (all parallel).
   7. SA acceptance on the full iteration's cost delta (Boltzmann, geometric cooling with alpha = 0.9995, T0 = 2% of initial cost).
   8. Track global best: **lowest cost first**, vehicle count as tiebreaker.
   
   The rationale for utilization-based ejection (vs. the previous size-based approach): low-utilization routes have more slack in surrounding routes to absorb their customers, whereas small routes might contain high-demand customers that are hard to redistribute. Interleaving SA operators with each ejection lets the solver immediately clean up after redistributions, rather than deferring optimization to a separate phase.
7. **Final inter-route + intra-route optimization**. After the merged SA+RM loop, a final pass of all three inter-route operators (relocate, swap, 2-opt*) followed by full intra-route optimization (TSP-approx + 2-opt) polishes the best solution found. This recovers any cost degradation from the last ejection cycles.
8. **Verify and report**: check capacity and time-window feasibility for all routes, print route details and per-phase timing/cost/vehicle summary.

Distances are computed on-the-fly (Euclidean, `VRP::get_dist()`), not precomputed into a matrix.

## Requirements

- `g++` with C++17 support
- OpenMP (`-fopenmp`) for the parallel build
- `make`, `bash`
- Target environment: Linux (HPC cluster with SLURM)

## Build

Sequential:
```bash
make
```

Parallel (adds `-fopenmp -DUSE_PARALLEL`):
```bash
make par
```

Clean:
```bash
make clean
```

## Run a Single Instance

```bash
./solve_cvrptw <instance_file> <angle_range> [sa_iterations]
```

- `angle_range` — angular width (in degrees) of each sweep cluster.
- `sa_iterations` — maximum merged SA+RM iterations (default: 1,000). Early stopping triggers if all routes reach 100% utilization.

Examples:

```bash
./solve_cvrptw testcase/C1_10_1.txt 180
./solve_cvrptw testcase/C1_10_1.txt 180 5000
```

## Batch Experiments

### `test.sh` — sweep angles and keep best

Runs the solver on every file in `testcase/` for each angle in the configured list (currently `180`), keeps the best result per instance.

```bash
# Sequential
bash test.sh

# Parallel
bash test.sh --parallel

# Custom SA iterations
bash test.sh --parallel --iterations 5000
```

Results go to `outputs/` (per-instance best output files) and `outputs/result.csv` (one summary line per instance).

### `run.sh` — quick sequential run

Builds sequential and runs all instances at a fixed angle of 30:

```bash
bash run.sh
bash run.sh --iterations 5000
```

### `test_huge.sh` — 10,000-customer XMLTW instances

Runs the solver on the 6 `XMLTW10000_*.txt` files in the project root (not `testcase/`). Same angle-sweep-and-keep-best logic as `test.sh`.

```bash
# Sequential
bash test_huge.sh

# Parallel
bash test_huge.sh --parallel

# Custom SA iterations
bash test_huge.sh --parallel --iterations 20000
```

Results go to `outputs/result_huge.csv`.

All scripts default to 10,000 SA iterations if `--iterations` is not specified (the solver binary itself defaults to 1,000 when no argument is given).

### SLURM (HPC cluster)

`submit_job.sh` — 1000-customer benchmarks, 4-hour wall time:

```bash
sbatch submit_job.sh
```

`submit_job_huge.sh` — 10,000-customer XMLTW instances, 12-hour wall time:

```bash
sbatch submit_job_huge.sh
```

Both use 1 node, 48 cores, partition `small`, `OMP_NUM_THREADS=48`.

## Output

Routes are printed to `stdout`. A summary line is written to `stderr` with per-phase metrics:

| Field | Description |
|-------|-------------|
| `File` | Input instance path |
| `Preprocessing_Time` | Clustering time (seconds) |
| `Construction_Time` | Clarke-Wright time (seconds) |
| `Construction_Cost` | Total distance after construction |
| `Construction_Vehicles` | Vehicle count after construction |
| `InterRoute_Time` | Inter-route optimization time (seconds) |
| `InterRoute_Cost` | Total distance after inter-route optimization |
| `InterRoute_Vehicles` | Vehicle count after inter-route optimization |
| `IntraRoute_Time` | Intra-route optimization time (seconds) |
| `IntraRoute_Cost` | Total distance after intra-route optimization |
| `IntraRoute_Vehicles` | Vehicle count after intra-route optimization |
| `SA_RM_Time` | Merged SA+RM phase time (seconds) |
| `SA_RM_Iterations` | Actual SA+RM iterations run (may be less than max if all routes hit 100% utilization) |
| `SA_RM_Cost` | Total distance after merged SA+RM |
| `SA_RM_Vehicles` | Vehicle count after merged SA+RM |
| `FinalOpt_Time` | Final inter+intra optimization time (seconds) |
| `Final_Cost` | Total distance of final solution |
| `Final_Vehicles` | Number of routes in the final solution |
| `Total_Time` | End-to-end wall time (seconds) |
| `route_length` | Length of the longest route (node count) |
| `VALID` | Printed only if all routes pass feasibility checks |

## Project Structure

```
solve_cvrptw.cpp          Main driver
Makefile                  Build rules (sequential / parallel)
submit_job.sh             SLURM job script (48 cores, 4h, 1000-customer benchmarks)
submit_job_huge.sh        SLURM job script (48 cores, 12h, 10,000-customer XMLTW)
test.sh                   Batch runner with angle sweep (testcase/)
test_huge.sh              Batch runner for XMLTW10000_*.txt files
run.sh                    Quick sequential batch runner

lib/
  vrp.h / vrp.cpp                     VRP data structures, instance parser, distance
  route_utils.h / route_utils.cpp     Route cost, feasibility checks, utilization stats, printing
  cluster/
    clustering.h / clustering.cpp     Angle-sweep and alternative clustering methods
  clark/
    clarke_wright.h / clarke_wright.cpp   Clarke-Wright savings heuristic (seq + parallel)
  optim/
    route_minimization.h / .cpp                    Vehicle count reduction (standalone, currently unused — logic merged into SA)
    sa_optimization.h / sa_optimization.cpp       Merged SA + route minimization loop
    intra_route_optimization.h / .cpp             Within-route: nearest-neighbor, 2-opt
    inter_route_optimization.h / .cpp             Between-route: relocate, swap, 2-opt*

XMLTW10000_*.txt          Gehring & Homberger 10,000-customer instances (6 files)
testcase/                 Gehring & Homberger benchmark instances (1000 customers)
outputs/                  Generated results from batch runs
```

Utility scripts (not part of the solver):
- `plot.py`, `plot_sa.py`, `plot_locations.py` — visualization
- `generator.py` — instance generation
- `sum_.py` — result aggregation
