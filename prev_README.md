# CVRPTW Solver

This project is a C++ implementation of a solver for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW). The current pipeline combines:

- clustering-based preprocessing,
- Clarke-Wright route construction,
- inter-route and intra-route local search,
- OpenMP-based parallelism in multiple stages.

The main executable is `solve_cvrptw`. The dataset to evaluate the algorithm is present in https://www.sintef.no/projectweb/top/vrptw/1000-customers/

## Project Overview

At a high level, the solver works as follows:

1. Read an instance from a file.
2. Build the distance graph for all nodes.
3. Create clusters using angle-sweep clustering.
4. Construct routes inside those clusters using Clarke-Wright.
5. Improve the routes with relocation, swap, 2-opt*, and post-processing steps.
6. Print the route set and summary metrics such as distance and runtime.


## Requirements

You need:

- `g++` with C++17 support
- OpenMP support (`-fopenmp`)
- `make`
- `bash`


This project is configured for Linux-style execution.

## Build

From the project root:

To build the executable(sequential version):

```bash
make
```
To build the executable(parallel version):

```bash
make par
```


To clean generated files:

```bash
make clean
```

## Run a Single Instance

The executable expects:

```bash
./solve_cvrptw <instance_file> <angle_range>
```

Example:

```bash
./solve_cvrptw c101.txt 45
```

Or with a file from the `testcase/` folder:

```bash
./solve_cvrptw testcase/c101.txt 45
```

## Run Batch Experiments

The repository also includes `test.sh`, which runs the solver on every file inside `testcase/` for multiple angle values and stores the best result.

Run (Sequential version):

```bash
bash test.sh
```

Run (Parallel version):

```bash
# bash test.sh -p
bash test.sh --parallel

```
 

The script currently tests these angles:
`30`,`45`,`60`,`90`,`180`

For each input instance, it:

1. runs the solver for each angle,
2. extracts the reported distance/cost,
3. keeps the best result,
4. writes outputs to the `outputs/` directory,
5. appends summary information to `outputs/result.csv`.

## Folder Structure

### `lib/`

Core source code for the solver.

### `lib/cluster/`

Clustering algorithms used before route construction.

- `clustering.cpp`: sweep-based clustering implementations, including parallel angle-sweep clustering
- `clustering.h`: clustering function declarations

### `lib/clark/`

Clarke-Wright savings-based route construction logic.

- `clarke_wright.cpp`: sequential and parallel Clarke-Wright variants
- `clarke_wright.h`: route construction API declarations

### `lib/optim/`

Route improvement operators.

- intra-route optimization
- inter-route relocation
- inter-route swap
- inter-route 2-opt*
- parallel optimization variants where implemented

### `testcase/`

Benchmark or input VRP instance files used for testing and experiments.

### `outputs/`

Generated results from batch runs.

Typical contents:

- best output files for each instance
- temporary logs during testing
- `result.csv` summary file

### Root Files

- `solve_cvrptw.cpp`: main program entry point
- `Makefile`: build rules
- `test.sh`: batch execution script


## Output

The program prints the generated clusters, total distance, final routes , and validation and timing statistics.

The summary line written to `stderr` includes values such as:

- preprocessing time
- route construction time
- post-optimization time
- initial cost
- intermediate optimization costs
- final cost
- total runtime
- number of vehicles used

## Example Workflow

```bash
make
./solve_cvrptw testcase/c101.txt 45
```
```bash
make par
./solve_cvrptw testcase/c101.txt 45
```

For batch evaluation:

```bash
bash test.sh
```
