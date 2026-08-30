import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    "Algorithm": [
        "cuOpt", 
        "C&W O(n³)", 
        "Sweep + C&W (v1)", 
        "Sweep + C&W (v2)"
    ],
    "Distance (Cost)": [35382.62, 47050.11667, 38760.7583, 38745.1083],
    "Total Time": [16761.20916667, 351.43, 8315.1097, 832.7398],
    "Vehicles Used": [58.91666667, 178.58333333, 70.5, 69.8333],
    "Avg Route Length": [34.66666667, 30.16666667, 31.4167, 32.1667]
}

df = pd.DataFrame(data)

# Setting up the figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Vehicle Routing Problem: Algorithm Performance Comparison', fontsize=16)

# List of columns to plot
metrics = ["Distance (Cost)", "Total Time", "Vehicles Used", "Avg Route Length"]
colors = ['skyblue', 'salmon', 'lightgreen', 'orange']

# Iterate through axes and metrics to create plots
for i, ax in enumerate(axes.flat):
    metric = metrics[i]
    ax.bar(df["Algorithm"], df[metric], color=colors[i])
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=15)
    
    # Adding data labels on top of bars
    for index, value in enumerate(df[metric]):
        ax.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()