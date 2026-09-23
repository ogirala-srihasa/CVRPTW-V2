#!/usr/bin/env python3
"""
solve_cuopt.py

Runs the NVIDIA cuOpt GPU routing solver on every CVRPTW instance in a
testcase folder and appends one row per instance to a CSV file.

Each instance is given the same wall-clock budget our own solver spent on
it, read from the Final_Time column of the solver's results CSV (written by
test.sh / run.sh). So if our pipeline took 45.2s on C1_10_1.txt, cuOpt is
given 45.2s on C1_10_1.txt -- an equal-budget comparison rather than an
arbitrary fixed one. Instances with no row in that CSV are skipped, since
there is nothing to match their budget to.

Instance files are expected in the standard Solomon VRPTW format:

    <instance name>

    VEHICLE
    NUMBER     CAPACITY
      <K>        <Q>

    CUSTOMER
    CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME

       0      40         50          0          0       1236          0
       1      45         68         10        912        967         90
       ...

This is the same format the project's own testcase/ files use (the same
Solomon 1000-customer VRPTW set referenced in README.md), so no conversion
is needed.

Usage:
    python3 solve_cuopt.py \
        --testcase_dir testcase \
        --results_csv outputs/result.csv \
        --output_csv outputs_cuopt/cuopt_results.csv

Requires the cuopt Python package (and cudf) to be installed and a CUDA
GPU to be visible to the process. See submit_cuopt_job.sh for how this is
wired up on the cluster.
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

# cuopt / cudf are only imported once we actually need them, so that
# --help and argument errors don't require a GPU environment.
def _import_cuopt():
    try:
        import cudf
        from cuopt import routing
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: could not import cudf / cuopt. Make sure you are running "
            "inside an environment with cuopt installed and a GPU available.\n"
            f"Original error: {exc}\n"
        )
        sys.exit(1)
    return cudf, routing


# Solver status codes, per cuOpt's routing.Assignment.get_status():
#   0 - SUCCESS, 1 - FAIL, 2 - TIMEOUT, 3 - EMPTY
STATUS_NAMES = {0: "SUCCESS", 1: "FAIL", 2: "TIMEOUT", 3: "EMPTY"}


def parse_solomon_instance(filepath):
    """Parse a Solomon-format VRPTW instance file.

    Returns a dict with:
        coords:   list of (x, y) tuples, index 0 is the depot
        demand:   list of int demands, index 0 is the depot (0)
        ready:    list of float ready times
        due:      list of float due times
        service:  list of float service times
        n_vehicles: int, fleet size given in the file
        capacity:   int, per-vehicle capacity given in the file
    """
    with open(filepath, "r") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    # Keep only non-blank lines for locating section headers, but remember
    # original content for the data rows.
    lines = [line.strip() for line in raw_lines if line.strip() != ""]

    if "VEHICLE" not in lines:
        raise ValueError("Could not find VEHICLE section")
    if "CUSTOMER" not in lines:
        raise ValueError("Could not find CUSTOMER section")

    veh_idx = lines.index("VEHICLE")
    # line after "VEHICLE" is the "NUMBER  CAPACITY" header,
    # the line after that holds the actual numbers
    veh_values = lines[veh_idx + 2].split()
    n_vehicles = int(veh_values[0])
    capacity = int(veh_values[1])

    cust_idx = lines.index("CUSTOMER")
    # line after "CUSTOMER" is the column header line; data starts after that
    data_start = cust_idx + 2

    coords, demand, ready, due, service = [], [], [], [], []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            _cust_no = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            dem = int(float(parts[3]))
            ready_t = float(parts[4])
            due_t = float(parts[5])
            serv_t = float(parts[6])
        except ValueError:
            # Not a data row (stray text, footer, etc.) - skip it.
            continue
        coords.append((x, y))
        demand.append(dem)
        ready.append(ready_t)
        due.append(due_t)
        service.append(serv_t)

    if len(coords) < 2:
        raise ValueError("Parsed fewer than 2 locations (depot + customers)")

    return {
        "coords": coords,
        "demand": demand,
        "ready": ready,
        "due": due,
        "service": service,
        "n_vehicles": n_vehicles,
        "capacity": capacity,
    }


def load_our_timings(results_csv):
    """Read our solver's results CSV and return its per-instance figures.

    Returns a dict keyed by instance basename (e.g. "C1_10_1.txt"), each
    value holding:
        time_s:   Final_Time, the end-to-end wall time our solver took.
                  This becomes cuOpt's time limit for that instance.
        cost:     Final_Cost, carried through so the output CSV is
                  self-contained for comparison.
        vehicles: Final_Vehicles, likewise.
        valid:    Valid (1/0), so a comparison against a run that failed
                  feasibility checks can be filtered out later.

    The batch scripts append to this file, so an instance run more than once
    appears more than once. The last row wins, i.e. the most recent run.
    """
    required = {"File", "Final_Time", "Final_Cost", "Final_Vehicles", "Valid"}

    with open(results_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{results_csv} does not look like a solver results CSV. "
                f"Expected at least the columns {sorted(required)}, found "
                f"{reader.fieldnames}. Older result files used a space-separated "
                "'Key: value' format with no header -- re-run test.sh to "
                "regenerate this file in the current CSV format."
            )

        timings = {}
        for row in reader:
            instance = os.path.basename((row.get("File") or "").strip())
            if not instance:
                continue
            try:
                time_s = float(row["Final_Time"])
            except (TypeError, ValueError):
                # Malformed row (truncated run, partial write) - ignore it
                # rather than handing cuOpt a garbage time limit.
                continue
            timings[instance] = {
                "time_s": time_s,
                "cost": row.get("Final_Cost", ""),
                "vehicles": row.get("Final_Vehicles", ""),
                "valid": row.get("Valid", ""),
            }

    return timings


def build_distance_matrix(coords):
    """Euclidean distance matrix as a list of lists (n x n)."""
    n = len(coords)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = d
            mat[j][i] = d
    return mat


def solve_instance(cudf, routing, instance, time_limit):
    """Solve one parsed instance with cuOpt at the given time limit (seconds).

    Returns a dict with status, status_name, total_cost, vehicles_used,
    solve_time_s. total_cost / vehicles_used are None if no solution was
    produced.
    """
    coords = instance["coords"]
    n_locations = len(coords)
    n_vehicles = instance["n_vehicles"]
    capacity = instance["capacity"]

    dist_matrix = build_distance_matrix(coords)
    cost_matrix = cudf.DataFrame(dist_matrix, dtype="float32")

    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(cost_matrix)
    data_model.add_transit_time_matrix(cost_matrix.copy(deep=True))

    demand_series = cudf.Series(instance["demand"])
    capacity_series = cudf.Series([capacity] * n_vehicles)
    data_model.add_capacity_dimension("demand", demand_series, capacity_series)

    earliest = cudf.Series(instance["ready"])
    latest = cudf.Series(instance["due"])
    data_model.set_order_time_windows(earliest, latest)
    data_model.set_order_service_times(cudf.Series(instance["service"]))

    depot_ready = instance["ready"][0]
    depot_due = instance["due"][0]
    vehicle_earliest = cudf.Series([depot_ready] * n_vehicles)
    vehicle_latest = cudf.Series([depot_due] * n_vehicles)
    data_model.set_vehicle_time_windows(vehicle_earliest, vehicle_latest)

    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(float(time_limit))

    t0 = time.perf_counter()
    solution = routing.Solve(data_model, solver_settings)
    elapsed = time.perf_counter() - t0

    status = solution.get_status()
    status_name = STATUS_NAMES.get(status, f"UNKNOWN({status})")

    total_cost = None
    vehicles_used = None
    if status in (0, 2):  # SUCCESS or TIMEOUT may still carry a solution
        try:
            total_cost = float(solution.get_total_objective())
            vehicles_used = int(solution.get_vehicle_count())
        except Exception:
            pass

    return {
        "status": status,
        "status_name": status_name,
        "total_cost": total_cost,
        "vehicles_used": vehicles_used,
        "solve_time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testcase_dir", default="testcase",
        help="Directory containing Solomon-format instance files (default: testcase)",
    )
    parser.add_argument(
        "--output_csv", default="outputs_cuopt/cuopt_results.csv",
        help="CSV file to append results to (default: outputs_cuopt/cuopt_results.csv)",
    )
    parser.add_argument(
        "--results_csv", default="outputs/result.csv",
        help="Our solver's results CSV. Each instance's cuOpt time limit is "
             "that instance's Final_Time (default: outputs/result.csv)",
    )
    args = parser.parse_args()

    try:
        timings = load_our_timings(args.results_csv)
    except FileNotFoundError:
        sys.stderr.write(
            f"ERROR: {args.results_csv} not found. Run our solver first "
            "(e.g. bash test.sh --parallel) so the per-instance time limits "
            "exist, then re-run this script.\n"
        )
        sys.exit(1)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)

    if not timings:
        sys.stderr.write(f"ERROR: no usable rows in {args.results_csv}\n")
        sys.exit(1)

    print(f"Loaded time limits for {len(timings)} instance(s) from "
          f"{args.results_csv}")

    cudf, routing = _import_cuopt()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)

    files = sorted(
        p for p in glob.glob(os.path.join(args.testcase_dir, "*"))
        if os.path.isfile(p)
    )
    if not files:
        sys.stderr.write(f"No files found in {args.testcase_dir}\n")
        sys.exit(1)

    # timeout_s is our solver's Final_Time for this instance. our_cost /
    # our_vehicles / our_valid come from the same row, so the output CSV can
    # be compared head-to-head without joining back to the results file.
    fieldnames = [
        "instance", "n_customers", "n_vehicles_available", "capacity",
        "timeout_s", "status", "status_name", "total_cost",
        "vehicles_used", "solve_time_s",
        "our_cost", "our_vehicles", "our_valid",
    ]

    skipped = []

    with open(args.output_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for filepath in files:
            instance_name = os.path.basename(filepath)

            # The time limit is our solver's wall time on this same instance,
            # so there is nothing to run if we never solved it.
            ours = timings.get(instance_name)
            if ours is None:
                print(f"[SKIP] {instance_name}: no row in {args.results_csv}")
                skipped.append(instance_name)
                continue

            timeout = ours["time_s"]
            if timeout <= 0:
                print(f"[SKIP] {instance_name}: Final_Time is {timeout}, "
                      "not a usable time limit")
                skipped.append(instance_name)
                continue

            try:
                instance = parse_solomon_instance(filepath)
            except Exception as exc:
                print(f"[SKIP] {instance_name}: failed to parse ({exc})")
                skipped.append(instance_name)
                continue

            n_customers = len(instance["coords"]) - 1
            print(f"=== {instance_name} "
                  f"({n_customers} customers, {instance['n_vehicles']} vehicles, "
                  f"capacity {instance['capacity']}) ===")

            row = {
                "instance": instance_name,
                "n_customers": n_customers,
                "n_vehicles_available": instance["n_vehicles"],
                "capacity": instance["capacity"],
                "timeout_s": timeout,
                "our_cost": ours["cost"],
                "our_vehicles": ours["vehicles"],
                "our_valid": ours["valid"],
            }
            try:
                result = solve_instance(cudf, routing, instance, timeout)
                row.update(result)
                print(
                    f"  timeout={timeout:.2f}s (our wall time)  "
                    f"status={result['status_name']:<8} "
                    f"cuopt_cost={result['total_cost']}  "
                    f"cuopt_vehicles={result['vehicles_used']}  "
                    f"our_cost={ours['cost']}  our_vehicles={ours['vehicles']}  "
                    f"wall_time={result['solve_time_s']:.2f}s"
                )
            except Exception as exc:
                row.update({
                    "status": "", "status_name": "ERROR", "total_cost": "",
                    "vehicles_used": "", "solve_time_s": "",
                })
                print(f"  timeout={timeout:.2f}s  ERROR: {exc}")

            writer.writerow(row)
            csvfile.flush()

    if skipped:
        print(f"\nSkipped {len(skipped)} instance(s) with no usable time "
              f"limit: {', '.join(skipped)}")

    print(f"\nDone. Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
