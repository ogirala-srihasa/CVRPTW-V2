import sys, random, math, os

# To plot the instance, uncomment the plotting lines near the bottom (it deteriorates performance)
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import matplotlib.pyplot as plt

if len(sys.argv) < 8:
    print('Missing arguments:\n\t python generate_vrptw.py n depotPos custPos demandType avgRouteSize instanceID randSeed '
          '[serviceTime] [horizon]')
    help = """

    n (number of customers)

    Depot positioning
        1 = Random
        2 = Centered
        3 = Cornered

    Customer positioning
        1 = Random
        2 = Clustered
        3 = Random-clustered

    Demand distribution
        1 = Unitary
        2 = Small, large var
        3 = Small, small var
        4 = Large, large var
        5 = Large, small var
        6 = Large, depending on quadrant
        7 = Few large, many small

    Average route size (also controls time-window tightness, Solomon-style:
    small values behave like the tight-window R1/C1 sets, large values like
    the wide-window R2/C2 sets)
        1 = Very short   (tight windows)
        2 = Short
        3 = Medium
        4 = Long
        5 = Very long
        6 = Ultra long    (wide windows)

    Optional arguments
        serviceTime  : fixed service time added at every customer (default 10)
        horizon      : planning horizon (end of depot's time window). If omitted,
                        it is computed automatically so that every customer's
                        window is guaranteed feasible for a depot round trip.

    Output: instance file XMLTW<n>_<depotPos><custPos><demandType><avgRouteSize>_<instanceID>.txt
            written in the classic Solomon (1987) VRPTW text format.

    For more details about the base generation process read:
        Uchoa et al (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. European Journal of Operational Research
        Queiroga, Eduardo, et al. (2022). 10,000 optimal CVRP solutions for testing machine learning based heuristics.
        """
    print(help)
    exit(0)


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


# constants
maxCoord = 1000
decay = 40

# read input arguments
n = int(sys.argv[1])
rootPos = int(sys.argv[2])
custPos = int(sys.argv[3])
demandType = int(sys.argv[4])
instanceID = int(sys.argv[6])
randSeed = int(sys.argv[7])  # random seed for reproducibility
if demandType > 7:
    print("Demand type out of range!")
    exit(0)

random.seed(randSeed)

nSeeds = random.randint(2, 6)

In = {1: (3, 5), 2: (5, 8), 3: (8, 12), 4: (12, 16), 5: (16, 25), 6: (25, 50)}
avgRouteSize = int(sys.argv[5])
if avgRouteSize > 6:
    print("Average route size out of range!")
    exit(0)
r = random.uniform(In[avgRouteSize][0], In[avgRouteSize][1])

# optional time-window related arguments
serviceTime = int(sys.argv[8]) if len(sys.argv) > 8 else 10
userHorizon = int(sys.argv[9]) if len(sys.argv) > 9 else None

# change '02d' if you need more than two digits (e.g. with '03d' you can index from 001 to 999)
instanceName = 'XMLTW' + str(n) + '_' + str(rootPos) + str(custPos) + str(demandType) + str(avgRouteSize) + '_' + format(instanceID, '02d')

pathToWrite = instanceName + '.txt'

depot = (-1, -1)  # depot position
S = set()  # set of coordinates for the customers

x_, y_ = (-1, -1)
# Root positioning
if rootPos == 1:
    x_ = random.randint(0, maxCoord)
    y_ = random.randint(0, maxCoord)
elif rootPos == 2:
    x_ = y_ = int(maxCoord / 2.0)
elif rootPos == 3:
    x_ = y_ = 0
else:
    print("Depot Positioning out of range!")
    exit(0)
depot = (x_, y_)

# Customer positioning
nRandCust = -1
if custPos == 3:
    nRandCust = int(n / 2.0)
elif custPos == 2:
    nRandCust = 0
elif custPos == 1:
    nRandCust = n
    nSeeds = 0
else:
    print("Customer Positioning out of range!")
    exit(0)

nClustCust = n - nRandCust

# Generating random customers
for i in range(1, nRandCust + 1):
    x_ = random.randint(0, maxCoord)
    y_ = random.randint(0, maxCoord)
    while (x_, y_) in S or (x_, y_) == depot:
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
    S.add((x_, y_))

nS = nRandCust

seeds = []
# Generation of the clustered customers
if nClustCust > 0:
    if nClustCust < nSeeds:
        print("Too many seeds!")
        exit(0)

    # Generate the seeds
    for i in range(nSeeds):
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
        S.add((x_, y_))
        seeds.append((x_, y_))
    nS = nS + nSeeds

    # Determine the seed with maximum sum of weights (w.r.t. all seeds)
    maxWeight = 0.0
    for i, j in seeds:
        w_ij = 0.0
        for i_, j_ in seeds:
            w_ij += 2 ** (-distance((i, j), (i_, j_)) / decay)
        if w_ij > maxWeight:
            maxWeight = w_ij

    norm_factor = 1.0 / maxWeight

    # Generate the remaining customers using Accept-reject method
    while nS < n:
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)

        weight = 0.0
        for i_, j_ in seeds:
            weight += 2 ** (-distance((x_, y_), (i_, j_)) / decay)
        weight *= norm_factor
        rand = random.uniform(0, 1)

        if rand <= weight:  # Will we accept the customer?
            S.add((x_, y_))
            nS = nS + 1

V = [depot] + list(S)  # set of vertices (index 0 = depot, matches Solomon's CUST NO. 0)

# Demands
demandMinValues = [1, 1, 5, 1, 50, 1, 51, 50, 1]
demandMaxValues = [1, 10, 10, 100, 100, 50, 100, 100, 10]
demandMin = demandMinValues[demandType - 1]
demandMax = demandMaxValues[demandType - 1]
demandMinEvenQuadrant = 51
demandMaxEvenQuadrant = 100
demandMinLarge = 50
demandMaxLarge = 100
largePerRoute = 1.5
demandMinSmall = 1
demandMaxSmall = 10

D = []  # demands (for customers 1..n, i.e. V[1:])
sumDemands = 0
maxDemand = 0

for i in range(2, n + 2):
    j = int((demandMax - demandMin + 1) * random.uniform(0, 1) + demandMin)
    if demandType == 6:
        if (V[i - 1][0] < maxCoord / 2.0 and V[i - 1][1] < maxCoord / 2.0) or (V[i - 1][0] >= maxCoord / 2.0 and V[i - 1][1] >= maxCoord / 2.0):
            j = int((demandMaxEvenQuadrant - demandMinEvenQuadrant + 1) * random.uniform(0, 1) + demandMinEvenQuadrant)
    if demandType == 7:
        if i < (n / r) * largePerRoute:
            j = int((demandMaxLarge - demandMinLarge + 1) * random.uniform(0, 1) + demandMinLarge)
        else:
            j = int((demandMaxSmall - demandMinSmall + 1) * random.uniform(0, 1) + demandMinSmall)
    D.append(j)
    if j > maxDemand:
        maxDemand = j
    sumDemands = sumDemands + j

if demandType != 6:
    random.shuffle(D)

# Generate capacity (same rule as the CVRP generator)
if sumDemands == n:
    capacity = math.floor(r)
else:
    capacity = max(maxDemand, math.ceil(r * sumDemands / n))
capacity = int(capacity)

# ---------------------------------------------------------------------------
# Time-window generation
# ---------------------------------------------------------------------------
# Window tightness is tied to avgRouteSize, mirroring how the Solomon/Homberger
# families pair short routes with tight windows (R1/C1) and long routes with
# wide windows (R2/C2).
tightnessByRouteSize = {
    1: (0.08, 0.18),
    2: (0.12, 0.25),
    3: (0.20, 0.40),
    4: (0.35, 0.60),
    5: (0.55, 0.85),
    6: (0.75, 1.00),
}
minWidthFrac, maxWidthFrac = tightnessByRouteSize[avgRouteSize]

custCoords = V[1:]  # coordinates for customers 1..n
depotDists = [distance(depot, c) for c in custCoords]

# Horizon must be large enough for the farthest customer to be visited and
# still let the vehicle return to the depot. Auto-compute if not provided.
minFeasibleHorizon = int(math.ceil(2 * max(depotDists) + serviceTime)) if custCoords else int(2 * maxCoord)
if userHorizon is not None:
    horizon = max(userHorizon, minFeasibleHorizon)
else:
    # Extra headroom (not just the bare minimum) so windows have room to vary
    horizon = int(math.ceil(minFeasibleHorizon * 1.3))

readyTimes = [0]  # depot
dueTimes = [horizon]  # depot
serviceTimes = [0]  # depot

for d0i in depotDists:
    earliest = math.ceil(d0i)
    latest = math.floor(horizon - d0i - serviceTime)
    if latest < earliest:
        latest = earliest  # degenerate but still feasible (zero-width window)

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

# ---------------------------------------------------------------------------
# Fleet size: one vehicle per customer (guarantees enough vehicles regardless
# of capacity/time-window tightness)
# ---------------------------------------------------------------------------
K = n

# ---------------------------------------------------------------------------
# Write the instance in Solomon (1987) VRPTW text format
# ---------------------------------------------------------------------------
Dfull = [0] + D  # demand list aligned with V (index 0 = depot)

f = open(pathToWrite, 'w')
f.write(instanceName + '\n\n')
f.write('VEHICLE\n')
f.write('NUMBER     CAPACITY\n')
f.write('{:>5}{:>13}\n\n'.format(K, capacity))
f.write('CUSTOMER\n')
f.write('CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME\n\n')

for i, v in enumerate(V):
    f.write('{:>4}{:>9}{:>9}{:>8}{:>12}{:>10}{:>13}\n'.format(
        i, v[0], v[1], Dfull[i], readyTimes[i], dueTimes[i], serviceTimes[i]
    ))

f.close()

print('Instance written to ' + pathToWrite)
print('n=%d, capacity=%d, vehicles=%d, horizon=%d, sumDemands=%d' % (n, capacity, K, horizon, sumDemands))

# x = [v[0] for v in V]
# y = [v[1] for v in V]
# x_s = [v[0] for v in seeds]
# y_s = [v[1] for v in seeds]
# plt.figure(figsize=(20, 20), dpi=80)
# plt.scatter(x, y, marker='o', color='blue', edgecolor='blue', s=40)
# plt.scatter(x_s, y_s, marker='o', color='magenta', edgecolor='magenta', s=40)
# plt.scatter([x[0]], [y[0]], marker='s', edgecolor='black', color='yellow', s=200)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(instanceName + '.png')
# plt.close()