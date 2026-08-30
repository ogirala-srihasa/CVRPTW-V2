"""
Plot and save customer/depot locations for every testcase in the testcase/ folder.

Usage:
    python plot_locations.py

Saves one PNG per testcase to the project root directory.
"""

import os
import matplotlib.pyplot as plt


def parse_solomon_file(filepath):
    """
    Parse a Solomon-format CVRPTW test case file.

    Returns:
        instance_name (str): Name from the first line of the file.
        depot (tuple): (x, y) coordinates of the depot (customer 0).
        customers (list of tuples): [(x, y), ...] for all non-depot customers.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # First line is the instance name
    instance_name = lines[0].strip()

    depot = None
    customers = []

    # Customer data starts after the header block.
    # Format:  CUST_NO  XCOORD  YCOORD  DEMAND  READY_TIME  DUE_DATE  SERVICE_TIME
    # The data lines begin after line 9 (0-indexed), i.e. from index 9 onward.
    for line in lines[9:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue

        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])

        if cust_id == 0:
            depot = (x, y)
        else:
            customers.append((x, y))

    return instance_name, depot, customers


def plot_and_save(instance_name, depot, customers, save_path):
    """
    Plot depot and customer locations and save to a PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot customers
    if customers:
        cx, cy = zip(*customers)
        ax.scatter(cx, cy, c="steelblue", s=20, alpha=0.7, label="Customers", zorder=2)

    # Plot depot
    if depot:
        ax.scatter(
            depot[0], depot[1],
            c="red", s=150, marker="*", edgecolors="black",
            linewidths=0.8, label="Depot", zorder=3,
        )

    ax.set_title(f"{instance_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcase_dir = script_dir
    output_dir = script_dir  # Save plots to project root

    if not os.path.isdir(testcase_dir):
        print(f"Error: testcase directory not found at {testcase_dir}")
        return

    # Gather all .txt files and sort them
    testcase_files = sorted(
        f for f in os.listdir(testcase_dir)
        if f.endswith(".txt")
    )

    if not testcase_files:
        print("No .txt testcase files found.")
        return

    print(f"Found {len(testcase_files)} testcase(s). Plotting...\n")

    for filename in testcase_files:
        filepath = os.path.join(testcase_dir, filename)
        instance_name, depot, customers = parse_solomon_file(filepath)

        # Save as <instance_name>.png
        save_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(output_dir, save_name)

        plot_and_save(instance_name, depot, customers, save_path)

    print(f"\nDone. {len(testcase_files)} plot(s) saved to: {output_dir}")


if __name__ == "__main__":
    main()
