import sys, os, glob, math, random

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python generate_vrptw_from_files.py [inputDir] [outputDir] [serviceTime] [seed]
#
#   inputDir     : folder containing CVRPLIB-style .vrp files (default: filo2_testcases)
#   outputDir    : folder to write Solomon-format .txt files to (default: filo2_testcases_vrptw)
#   serviceTime  : fixed service time added at every customer (default: 10)
#   seed         : base random seed; each file gets seed + its index for reproducibility (default: 1)
#
# For every "<name>.vrp" file found in inputDir, this script:
#   1. Parses NAME / CAPACITY / DIMENSION / NODE_COORD_SECTION / DEMAND_SECTION / DEPOT_SECTION.
#   2. Infers a Solomon-style time-window "tightness" class (narrow <-> wide) from the
#      instance's own CAPACITY and average customer demand -- there's no explicit
#      avgRouteSize field on a real instance, so we estimate the equivalent
#      "average customers per route" as capacity / average demand and bucket it the
#      same way the synthetic CVRP generator does.
#   3. Generates a ready_time/due_date window per customer that is guaranteed feasible
#      for a depot round trip (arrive after depot->customer travel time, leave in time
#      to be back at the depot by the horizon).
#   4. Sets the fleet size (NUMBER) equal to the number of customers, so capacity/time
#      windows are never the bottleneck on available vehicles.
#   5. Writes "<name>.txt" in the classic Solomon (1987) VRPTW text format.

inputDir = sys.argv[1] if len(sys.argv) > 1 else 'filo2_testcases'
outputDir = sys.argv[2] if len(sys.argv) > 2 else 'filo2_testcases_vrptw'
serviceTime = int(sys.argv[3]) if len(sys.argv) > 3 else 10
baseSeed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

os.makedirs(outputDir, exist_ok=True)

# Same bucket boundaries used by the synthetic CVRP generator's avgRouteSize argument,
# and the same tightness ranges used by generate_vrptw.py, so real and synthetic
# instances stay on a comparable scale.
routeSizeBuckets = [(5, 1), (8, 2), (12, 3), (16, 4), (25, 5), (float('inf'), 6)]
tightnessByRouteSize = {
    1: (0.08, 0.18),
    2: (0.12, 0.25),
    3: (0.20, 0.40),
    4: (0.35, 0.60),
    5: (0.55, 0.85),
    6: (0.75, 1.00),
}


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def bucket_for_route_size(r):
    for threshold, bucket in routeSizeBuckets:
        if r < threshold:
            return bucket
    return 6


def parse_cvrp(path):
    name = None
    capacity = None
    dimension = None
    coords = {}   # id -> (x, y)
    demands = {}  # id -> demand
    depot_id = None
    section = None

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith('NAME'):
                name = line.split(':', 1)[1].strip()
            elif upper.startswith('CAPACITY'):
                capacity = int(line.split(':', 1)[1].strip())
            elif upper.startswith('DIMENSION'):
                dimension = int(line.split(':', 1)[1].strip())
            elif upper.startswith('NODE_COORD_SECTION'):
                section = 'coord'
            elif upper.startswith('DEMAND_SECTION'):
                section = 'demand'
            elif upper.startswith('DEPOT_SECTION'):
                section = 'depot'
            elif upper.startswith('EOF'):
                break
            elif section == 'coord':
                parts = line.split()
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
            elif section == 'demand':
                parts = line.split()
                idx = int(parts[0])
                demands[idx] = int(parts[1])
            elif section == 'depot':
                parts = line.split()
                val = int(parts[0])
                if val != -1 and depot_id is None:
                    depot_id = val

    if depot_id is None:
        depot_id = 1  # sensible default if DEPOT_SECTION is missing/empty

    return name, capacity, dimension, coords, demands, depot_id


def convert_file(path, fileIndex):
    name, capacity, dimension, coords, demands, depot_id = parse_cvrp(path)
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    if capacity is None or not coords:
        print('Skipping %s: could not parse CAPACITY / coordinates' % path)
        return

    random.seed(baseSeed + fileIndex)

    ids = sorted(coords.keys())
    if depot_id not in coords:
        depot_id = ids[0]
    custIds = [i for i in ids if i != depot_id]
    n = len(custIds)
    if n == 0:
        print('Skipping %s: no customers found' % path)
        return

    depot = coords[depot_id]
    custCoords = [coords[i] for i in custIds]
    D = [demands.get(i, 0) for i in custIds]

    sumDemands = sum(D)
    avgDemand = sumDemands / float(n) if n > 0 else 1.0
    avgDemand = max(avgDemand, 1e-9)
    estRouteSize = capacity / avgDemand
    routeSizeBucket = bucket_for_route_size(estRouteSize)
    minWidthFrac, maxWidthFrac = tightnessByRouteSize[routeSizeBucket]

    depotDists = [distance(depot, c) for c in custCoords]
    maxDist = max(depotDists)
    minFeasibleHorizon = int(math.ceil(2 * maxDist + serviceTime))
    horizon = int(math.ceil(minFeasibleHorizon * 1.3))

    readyTimes = [0]
    dueTimes = [horizon]
    serviceTimes = [0]

    for d0i in depotDists:
        earliest = math.ceil(d0i)
        latest = math.floor(horizon - d0i - serviceTime)
        if latest < earliest:
            latest = earliest

        span = latest - earliest
        width = int(round(span * random.uniform(minWidthFrac, maxWidthFrac)))
        width = max(width, 0)

        if span > 0:
            center = random.uniform(earliest, latest)
        else:
            center = earliest

        ready = max(earliest, int(math.floor(center - width / 2.0)))
        due = min(latest, int(math.ceil(center + width / 2.0)))
        if due < ready:
            due = ready

        readyTimes.append(ready)
        dueTimes.append(due)
        serviceTimes.append(serviceTime)

    K = n  # fleet size equal to the number of customers

    outPath = os.path.join(outputDir, name + '.txt')
    with open(outPath, 'w') as f:
        f.write(name + '\n\n')
        f.write('VEHICLE\n')
        f.write('NUMBER     CAPACITY\n')
        f.write('{:>5}{:>13}\n\n'.format(K, capacity))
        f.write('CUSTOMER\n')
        f.write('CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME\n\n')

        allCoords = [depot] + custCoords
        allDemands = [0] + D
        lines = []
        for i in range(len(allCoords)):
            x, y = allCoords[i]
            xOut = int(x) if float(x).is_integer() else x
            yOut = int(y) if float(y).is_integer() else y
            lines.append('{:>4}{:>10}{:>10}{:>8}{:>12}{:>10}{:>13}\n'.format(
                i, xOut, yOut, allDemands[i], readyTimes[i], dueTimes[i], serviceTimes[i]
            ))
        f.writelines(lines)

    print('%s -> %s  (n=%d, capacity=%d, vehicles=%d, horizon=%d, routeSizeBucket=%d)' % (
        os.path.basename(path), outPath, n, capacity, K, horizon, routeSizeBucket))


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(inputDir, '*.vrp')))
    if not files:
        print('No .vrp files found in %s' % inputDir)
        sys.exit(0)
    for idx, path in enumerate(files):
        convert_file(path, idx)