import sys
import re
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_sa.py <sa_log_file>")
    print("  Generate the log: ./solve_cvrptw testcase/C1_10_1.txt 45 2> sa_log.txt")
    sys.exit(1)

iters, current, best = [], [], []

with open(sys.argv[1]) as f:
    for line in f:
        if not line.startswith("SA_ITER"):
            continue
        m = re.search(r'SA_ITER\s+(\d+)\s+current=([\d.]+)\s+best=([\d.]+)', line)
        # m = re.search(r'SA_ITER\s+(\d+)\s+current=([0-9\.eE\+\-]+)\s+best=([0-9\.eE\+\-]+)', line)
        if m:
            iters.append(int(m.group(1)))
            current.append(float(m.group(2)))
            best.append(float(m.group(3)))

if not iters:
    print("No SA_ITER lines found in", sys.argv[1])
    sys.exit(1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(iters, current, label="Current cost", alpha=0.6)
ax.plot(iters, best, label="Best cost", linewidth=2)
ax.set_xlabel("SA Iteration")
ax.set_ylabel("Cost")
ax.set_title("SA Post-Optimization Convergence")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sa_convergence.png", dpi=150)
plt.show()
print("Saved sa_convergence.png")
