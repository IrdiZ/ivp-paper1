import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("experiment_results.csv")

# Extract parameter columns for easier access
block_sizes = [3, 5, 7, 11]
disparity_values = [128, 256, 512]

# Prepare long-form data for visualization
long_data = []
for block_size in block_sizes:
    for disparity in disparity_values:
        long_data.append(
            data[["folder", 
                  f"block_{block_size}_disparities_{disparity}_mae", 
                  f"block_{block_size}_disparities_{disparity}_time"]]
            .rename(columns={
                f"block_{block_size}_disparities_{disparity}_mae": "MAE",
                f"block_{block_size}_disparities_{disparity}_time": "Time"
            })
            .assign(BlockSize=block_size, Disparities=disparity)
        )

long_data = pd.concat(long_data)

# Create visualizations
sns.set(style="whitegrid", palette="muted", font_scale=1.5)

# 1. MAE vs. Time Trade-off
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=long_data,
    x="Time",
    y="MAE",
    hue="BlockSize",
    style="Disparities",
    size="Disparities",
    sizes=(50, 200),
    alpha=0.8
)
plt.title("MAE vs. Execution Time")
plt.xlabel("Execution Time (s)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend(title="Legend")
plt.tight_layout()
plt.savefig("mae_vs_time_tradeoff.png")
plt.show()

# 2. Bar Plot: Average MAE for Each Block Size and Disparity
avg_mae = long_data.groupby(["BlockSize", "Disparities"])["MAE"].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(
    data=avg_mae,
    x="BlockSize",
    y="MAE",
    hue="Disparities",
    palette="muted"
)
plt.title("Average MAE by Block Size and Disparities")
plt.xlabel("Block Size")
plt.ylabel("Average MAE")
plt.legend(title="Disparities")
plt.tight_layout()
plt.savefig("avg_mae_by_blocksize_disparities.png")
plt.show()

# 3. Heatmap: MAE vs. Execution Time (Summarized)
heatmap_data = long_data.pivot_table(
    index="BlockSize", columns="Disparities", values="MAE", aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": "MAE"}
)
plt.title("Heatmap of MAE by Block Size and Disparities")
plt.xlabel("Disparities")
plt.ylabel("Block Size")
plt.tight_layout()
plt.savefig("mae_heatmap.png")
plt.show()

# 4. Identify Best Approach
# Determine the best-performing configuration for lowest MAE and Time
best_config = long_data.loc[
    long_data.groupby(["BlockSize", "Disparities"])["MAE"].idxmin()
]

print("Best Performing Configurations:")
print(best_config.sort_values(by=["MAE", "Time"]).reset_index(drop=True))

# Save summarized results to CSV
best_config.to_csv("best_performing_configurations.csv", index=False)

# Additional Insights
print("\nInsights:")
print(f"Lowest Overall MAE: {best_config['MAE'].min():.3f}")
print(f"Fastest Execution Time: {best_config['Time'].min():.2f} seconds")
