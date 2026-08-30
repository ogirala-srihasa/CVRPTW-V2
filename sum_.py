import re
from collections import defaultdict

filename = "outputs/result.csv"
group_size = 12

groups = []
current = defaultdict(float)
count = 0

with open(filename) as f:
    for line in f:
        pairs = re.findall(r'(\w+):\s*([\d.]+)', line)

        for key, val in pairs:
            current[key] += float(val)

        count += 1

        if count % group_size == 0:
            avg = {k: v/group_size for k, v in current.items()}
            groups.append(avg)
            current = defaultdict(float)

for i, g in enumerate(groups, 1):
    print(f"\nAlgorithm {i} (Average of {group_size} instances)")
    for k, v in g.items():
        print(f"{k}: {v:.4f}")