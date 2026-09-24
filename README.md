# CVRPTW Solver

A C++ solver for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW), parallelized with OpenMP. Targets the [Gehring & Homberger benchmark instances](https://www.sintef.no/projectweb/top/vrptw/1000-customers/) and Italian province instances ranging from 20K to 1M customers.

## Solver Pipeline

The solver runs three phases. The central idea is that **all expensive local search happens inside fixed-size clusters**, where the route count per cluster is small, and only two comparatively cheap global passes run over the whole instance afterwards.

**0. Read and cluster.** The instance is parsed from a Solomon-format file (`lib/vrp.cpp`). `clustering_polar_fixed_size` (`lib/cluster/clustering.cpp`) sorts customers by polar angle from the depot and chunks them into fixed-size groups (default 1000, set by `argv[2]`). There is no random start index — clustering is deterministic, so phase-to-phase comparisons are reproducible across runs apart from the SA RNG.

**1. Construction phase** (`construction_phase`, `lib/optim/pipeline.cpp`). Each cluster independently runs the full optimization stack:

1. **Clarke-Wright savings** (`lib/clark/clarke_wright.cpp`) — one route per customer, then repeated best-merge over route pairs. Four orientations per pair (forward/forward, reversed/forward, forward/reversed, and the swapped order), scored as `0.7 * dist_saving - 0.3 * waiting`, with every candidate re-verified against capacity and time windows before acceptance.
2. **Inter-route optimization** (`lib/optim/inter_route_optimization.cpp`) — relocate, swap, and 2-opt*, each best-improvement with a full restart after every accepted move. Also erases routes that decay to `[DEPOT, DEPOT]`.
3. **Intra-route optimization** (`lib/optim/intra_route_optimization.cpp`) — nearest-neighbor TSP approximation followed by 2-opt, keeping whichever of the two orderings is cheaper per route. DEPOT bookends are stripped before this phase (`postProcessIt` reorders all route elements) and re-added afterward.
4. **Vehicle-count minimization** (`route_min_v2`, `lib/optim/route_minimization.cpp`) — sorts routes by load ascending, then repeatedly ejects the emptiest route and reinserts its customers at their cheapest feasible positions. An ejection is committed only if *every* customer finds a feasible slot; the first leftover rolls that ejection back and ends the loop. Self-terminating, so it takes no iteration budget.
5. **SA-only** (`sa_only_optimization`) — 10,000 iterations of ruin-and-recreate (14 random removals + cheapest feasible reinsertion) plus the same four operators, with early stopping via stagnation (<0.1% improvement over a 300-iteration window) or temperature floor (`T < 1e-5 * cost`).

**2. Merge phase** (`merge_phase`, `lib/optim/pipeline.cpp`). Clustering necessarily cuts through natural customer groups, so this phase merges routes back across those cuts using `clarke_wright_merge_routes` — the same savings machinery as construction, but seeded from the **existing routes** rather than from singletons, so the ordering computed inside each route survives and only whole-route concatenations are considered. Scoping depends on size (see Design Notes).

**3. Post-merge optimization** (`post_merge_optimization`, `lib/optim/sa_optimization.cpp`). 1,000 iterations of: sort routes by utilization, eject the two least-utilized, reinsert their customers at the cheapest feasible position (parallel search, tightest time windows first), run Clarke-Wright on anything that found no feasible slot, then SA acceptance on the iteration delta. This loop deliberately carries **no local-search operators**.

**4. Verify and report.** Capacity and time-window feasibility are checked for every route; routes go to `stdout` and one CSV row to `stderr`.

Distances are computed on the fly (Euclidean, `VRP::get_dist()`), never precomputed into a matrix — an N×N matrix is not an option at 1M customers.

## Design Notes

Why the pipeline is shaped this way, for anyone extending it later.

### Why cluster-local optimization

The inter-route and SA operators are O(R²·L²) in the number of routes R. Run globally they are fine at 1,000 customers and unusable at 1M. Confining them to a 1000-customer cluster caps R at roughly 30–50 regardless of instance size, which is what makes the large instances tractable at all. The merge and post-merge phases exist to recover the quality lost at cluster boundaries.

### Parallelization strategy

Two modes, chosen once from the cluster count:

| Condition | Outer loop | Inner phases |
|---|---|---|
| `num_clusters >= 2 * omp_get_max_threads()` | `parallel for` over clusters | sequential variants |
| otherwise | sequential over clusters | `_parallel` variants, full thread team |

The `2×` factor is tail balance — at 50 clusters on 48 threads the outer form runs two waves at roughly 50% efficiency, so the inner-parallel form wins there. The crossover lands around 48K–96K customers and is a single constant in `construction_phase`.

In the outer-parallel form, iteration `c` writes only `grouped_routes[c]`, so there is no race and no critical section. The SA entry points carry their own `omp parallel` regions; inside the outer loop those become nested regions and run single-threaded. `main` calls `omp_set_max_active_levels(1)` to make that an invariant rather than a default being relied on.

> **`clarke_wright_cvrptw_parallel_v3` must not be used in this pipeline.** It calls `omp_set_max_active_levels(2)`, which would turn the outer 48 threads into 48×48.

### Why the merge phase is scoped by route count

Clarke-Wright rescans every remaining route pair after every accepted merge, so merging R routes costs roughly R³/6 pair evaluations, each constructing four candidate routes and re-verifying time windows. On 48 threads that is sub-second at R=500, a few minutes at R=2000, and roughly an hour at R=5000. Expected route counts run ~250 at 10K customers, ~500 at 20K, ~2000 at 80K, and ~25000 at 1M.

So `merge_phase` takes one of two paths, split at `GLOBAL_MERGE_ROUTE_LIMIT = 2000` (defined in `solve_cvrptw.cpp`):

- **Global merge** — every route against every other. Simple, and what every Solomon instance and the smaller provinces get.
- **Adjacent-boundary merge** — two rounds of disjoint pair merges over neighbouring angular slices: round 0 takes boundaries (0,1) (2,3) …, round 1 takes (1,2) (3,4) …, plus the circular wraparound boundary (K−1, 0), which neither round covers for either parity of K. Within a round the pairs touch disjoint groups, so the `parallel for` is race-free and needs no critical section; the inner merge is the sequential variant because parallelism already lives at the pair level.

Restricting to neighbours costs almost nothing in solution quality: the savings term `d(0,i) + d(0,j) − d(i,j)` is only positive when i and j are close as seen from the depot, so merges between non-adjacent angular slices would be evaluated and rejected anyway. After each pair merge, a merged route is assigned to whichever group its first customer originally belonged to — that split only decides which pairs the next round considers and cannot affect feasibility.

The known limitation is that with two rounds a route migrates at most one slice per round, so a route that ought to move two slices over will not. The post-merge phase, which reinserts ejected customers anywhere in the solution, is the mechanism that catches those.

### Why the post-merge loop has no local-search operators

It runs on the whole instance, where relocate / swap / 2-opt* are exactly the O(R²·L²) scans that motivated cluster-local optimization in the first place. All of that work has already been done inside the clusters. What remains genuinely useful at global scope is ejection and reinsertion, whose cost is driven by the ~20–50 ejected customers rather than by R.

That absence is also why the loop's **100% utilization guard is keyed on `routes[1]`, not `routes[0]`**. After the ascending load sort, `routes[0]` and `routes[1]` are the two least-utilized routes, so if `routes[1]` is already at capacity then no pair in the solution can be ejected. Keying the check on `routes[0]` instead would let the walking ejection index reset to 0 and spin — and with no operators to fall back on, each spin is a full solution copy plus a load sort that change nothing. `sa_post_optimization` can afford the weaker check because its operators still run on a skipped iteration.

### Why route_min_v2 replaced the SA+RM loop

The construction phase originally ran `sa_post_optimization` here: 10,000 iterations of
ejection *plus* relocate / swap / 2-opt* / intra-2-opt *plus* SA acceptance, tracking its
best solution **by cost**. That was the single most expensive thing in the pipeline.

`route_min_v2` asks one question instead — can this vehicle be removed while keeping every
route feasible — and answers it in at most R passes rather than 10,000 iterations. It has
no operators, no SA acceptance, and no cost check at all.

The trade is explicit: **fewer vehicles, higher distance** out of this phase, since
absorbing another route's customers lengthens whatever takes them and nothing gates that.
`sa_only_optimization` still runs afterward and claws distance back, but the local search
that used to be interleaved with each ejection is gone from this stage.

Stopping at the first failed ejection is a heuristic, not a proof that no route is
removable — a fuller route further down the ordering could have looser time windows and
still be dissolvable. Continuing past the failure would eliminate more vehicles at
proportionally more search cost.

`sa_post_optimization` remains in `sa_optimization.cpp`, uncalled, so the previous
behavior is one line away in `construct_cluster_routes`.

### Cooling schedules

`sa_only_optimization` uses a fixed `alpha = 0.9995`, tuned for its 10,000-iteration budget (as does `sa_post_optimization`, though the pipeline no longer calls it). `post_merge_optimization` instead derives `alpha = pow(1e-3, 1.0 / max_iterations)` so temperature reaches 0.1% of T0 for any budget; at 1,000 iterations the fixed 0.9995 only reaches 0.6·T0, which is a near-constant-temperature random walk. All three use `T0 = 2%` of the phase's initial cost.

### Known limitations

- **Clarke-Wright is the dominant cost.** The merge loop is ~R³ and reallocates four candidate route vectors per pair. Moving to a classic savings-list formulation (build the list once, sort descending, single pass) would be O(R² log R) instead, and is the single largest remaining speedup available.
- **Feasibility checking truncates time.** `tw_t` is `unsigned int`, so `verify_single_route` / `verify_route` accumulate arrival times in integer arithmetic and truncate each leg, and `Point::earlyTime/latestTime/serviceTime` truncate the file's doubles on read. Truncation biases arrival times downward, making the final check optimistic. Clarke-Wright uses its own double-precision `compute_arrival_time`, so the two feasibility notions disagree slightly.
- **Route cost is cached.** `calculate_route_distance` sums `RouteNode::dist_from_prev`, so any route mutation not followed by `recalculate_pred_distances` silently corrupts the reported cost.
- **Alternative clustering methods** (`clustering_hierarchical`, `clustering_kmeans_plus_plus`, `clustering_k_far`, `clustering_kmedoid`) and the angle-sweep functions remain in `clustering.cpp` but are no longer called by the driver.

## Requirements

- `g++` with C++17 support
- OpenMP (`-fopenmp`) for the parallel build
- `make`, `bash`
- Target environment: Linux (HPC cluster with SLURM)

## Build

```bash
make          # sequential
make par      # parallel, adds -fopenmp -DUSE_PARALLEL
make clean
```

## Run a Single Instance

```bash
./solve_cvrptw <instance_file> <cluster_size> [sa_rm_iterations] [sa_only_iterations] [post_opt_iterations]
#                                                  ^ accepted but unused
```

- `cluster_size` — customers per cluster (1000 is the tuned default).
- `sa_rm_iterations` — **unused.** The construction phase now minimizes vehicles with `route_min_v2`, which runs until an ejection fails rather than for a fixed count. The argument is still parsed so existing scripts and queued jobs keep their positions.
- `sa_only_iterations` — per-cluster SA-only iterations (default 10,000). Stops early on stagnation or temperature floor.
- `post_opt_iterations` — global post-merge iterations (default 1,000). Stops early once every route is at full capacity.

Examples:

```bash
./solve_cvrptw testcase/C1_10_1.txt 1000
./solve_cvrptw testcase/C1_10_1.txt 1000 5000 5000 500
```

## Batch Experiments

All three batch scripts take the same flags and write a real CSV (header plus one row per instance).

```bash
bash test.sh [--parallel] [--cluster-size N] [--sa-rm-iterations N] \
             [--sa-iterations N] [--post-opt-iterations N]
```

| Script | Instances | Results |
|---|---|---|
| `test.sh` | `testcase/*` (1,000-customer benchmarks) | `outputs/result.csv` |
| `test_huge.sh` | `XMLTW10000_*.txt` in the project root | `outputs/result_huge.csv` |
| `test_i_instances.sh` | `I_testcases/*.txt` (20K–1M customers) | `outputs/result_i_instances.csv` |
| `run.sh` | `testcase/*`, sequential build, no flags for parallel | `outputs/result.csv` |

Per-instance `stdout` goes to `outputs/<instance>.out`. Defaults are cluster size 1000, 10,000 SA-only, 1,000 post-opt (the SA+RM argument is accepted and ignored).

### SLURM (HPC cluster)

```bash
sbatch submit_job.sh              # 1000-customer benchmarks, 4h
sbatch submit_job_huge.sh         # 10,000-customer XMLTW, 12h
sbatch submit_job_i_instances.sh  # Italian province instances, 24h
```

All SLURM jobs use 1 node, 48 cores, partition `small`, `OMP_NUM_THREADS=48`.

## Output

Routes and per-phase progress are printed to `stdout`. Exactly one CSV row is written to `stderr` as the last line; the batch scripts append it to the result file and write the header once.

| Column | Description |
|---|---|
| `File` | Input instance path |
| `Construction_Time` | Wall time of the per-cluster construction phase (seconds) |
| `Construction_Cost` | Total distance after construction, summed over clusters |
| `Construction_Vehicles` | Vehicle count after construction, summed over clusters |
| `Merge_Time` | Wall time of the Clarke-Wright merge phase (seconds) |
| `Merge_Cost` | Total distance after merging |
| `Merge_Vehicles` | Vehicle count after merging |
| `PostOpt_Time` | Wall time of the post-merge optimization phase (seconds) |
| `PostOpt_Cost` | Total distance after post-merge optimization |
| `PostOpt_Vehicles` | Vehicle count after post-merge optimization |
| `Final_Time` | End-to-end wall time (seconds), including file read and clustering |
| `Final_Cost` | Total distance of the final solution |
| `Final_Vehicles` | Number of routes in the final solution |
| `Valid` | `1` if every route passes capacity and time-window checks, else `0` |

`Final_Cost` and `Final_Vehicles` duplicate the post-opt values by construction; the gap between `Final_Time` and the three phase times is preprocessing. The row is emitted even when verification fails, so a bad run shows up as `Valid=0` rather than vanishing from the results file.

## cuOpt Baseline Comparison

`solve_cuopt.py` runs NVIDIA's cuOpt GPU routing solver over the same instances, as a
reference point for our own results. It parses the same Solomon files directly, so no
conversion step is needed.

The comparison is **equal-budget**: each instance gives cuOpt exactly the wall-clock time
our solver spent on that same instance, read from the `Final_Time` column of our results
CSV. If our pipeline took 45.2s on `C1_10_1.txt`, cuOpt gets 45.2s on `C1_10_1.txt`. A
fixed budget would have measured something else entirely — how each solver happens to
behave at an arbitrary time limit, rather than which one does more with the same time.

This means **our solver must be run first**; `run_cuopt.sh` exits with an error if the
results CSV is missing. Instances with no row in it are skipped and listed at the end,
rather than being given a substituted budget that would make the row a non-comparison.
When an instance appears more than once (the batch scripts append), the last row wins.

```bash
sbatch submit_job.sh          # produces outputs/result.csv
sbatch submit_cuopt_job.sh    # reads it, runs cuOpt at matched budgets
```

Or directly:

```bash
bash run_cuopt.sh                                             # testcase/ vs outputs/result.csv
bash run_cuopt.sh I_testcases outputs/result_i_instances.csv  # another instance set
```

Results append to `outputs_cuopt/cuopt_results.csv`, one row per instance:

| Column | Description |
|---|---|
| `instance` | Instance file basename, the key joining the two CSVs |
| `n_customers`, `n_vehicles_available`, `capacity` | Parsed from the instance file |
| `timeout_s` | Time limit given to cuOpt — our solver's `Final_Time` |
| `status`, `status_name` | cuOpt status (0 SUCCESS, 1 FAIL, 2 TIMEOUT, 3 EMPTY) |
| `total_cost`, `vehicles_used` | cuOpt's solution, blank if none was produced |
| `solve_time_s` | Measured cuOpt wall time, to confirm it respected the limit |
| `our_cost`, `our_vehicles`, `our_valid` | Our figures from the same row, so the file is self-contained |

`submit_cuopt_job.sh` requests the `gpu` partition with one GPU and an 8-hour limit, and
expects a venv at `~/cuopt_env` with the `cuopt-cu12` package — see the one-time setup
comments at the top of that script. The results CSV is validated *before* cuOpt is
imported, so a missing or stale-format file fails in seconds rather than after GPU init.

Two things to watch:

- Delete an existing `outputs_cuopt/cuopt_results.csv` before the first run after a
  column change. The script appends and only writes a header when the file is absent.
- `submit_cuopt_job.sh` does `cd "$SLURM_SUBMIT_DIR"`, so it resolves `outputs/result.csv`
  relative to wherever you submitted from. Submit both jobs from the same directory, or
  pass the path to `run_cuopt.sh` explicitly.

## Project Structure

```
solve_cvrptw.cpp          Main driver: read -> cluster -> 3 phases -> verify -> CSV row
Makefile                  Build rules (sequential / parallel)
submit_job.sh             SLURM job script (48 cores, 4h, 1000-customer benchmarks)
submit_job_huge.sh        SLURM job script (48 cores, 12h, 10,000-customer XMLTW)
submit_job_i_instances.sh SLURM job script (48 cores, 24h, Italian province instances)
test.sh                   Batch runner (testcase/)
test_huge.sh              Batch runner for XMLTW10000_*.txt files
test_i_instances.sh       Batch runner for I_testcases/*.txt
run.sh                    Quick sequential batch runner
generatefromvrp.py        Converts CVRPLIB .vrp files to Solomon VRPTW format

solve_cuopt.py            cuOpt GPU baseline; time limits read from our results CSV
run_cuopt.sh              Batch runner for the cuOpt baseline
submit_cuopt_job.sh       SLURM job script (1 GPU, 8h, gpu partition)

lib/
  vrp.h / vrp.cpp                     VRP data structures, instance parser, distance
  route_utils.h / route_utils.cpp     Route cost, feasibility checks, utilization stats, printing
  cluster/
    clustering.h / clustering.cpp     Fixed-size polar clustering (+ unused alternatives)
  clark/
    clarke_wright.h / clarke_wright.cpp   Savings heuristic: from singletons, and from existing routes
  optim/
    pipeline.h / pipeline.cpp                     Construction phase and merge phase orchestration
    sa_optimization.h / sa_optimization.cpp       SA-only and post-merge loops (SA+RM retained, uncalled)
    intra_route_optimization.h / .cpp             Within-route: nearest-neighbor, 2-opt
    inter_route_optimization.h / .cpp             Between-route: relocate, swap, 2-opt*
    route_minimization.h / .cpp                   route_min_v2 (used by the construction phase); minimize_routes (unused)

XMLTW10000_*.txt          Gehring & Homberger 10,000-customer instances (6 files)
testcase/                 Gehring & Homberger benchmark instances (1000 customers)
I_testcases/              Italian province CVRPTW instances (20K–1M customers, generated from .vrp)
outputs/                  Generated results from batch runs
outputs_cuopt/            Generated results from cuOpt baseline runs
```

Utility scripts (not part of the solver):
- `plot.py`, `plot_sa.py`, `plot_locations.py` — visualization
- `generator.py` — instance generation
- `sum_.py` — result aggregation
