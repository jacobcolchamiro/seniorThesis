import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Define the folder containing the CSV files
folder_path = "beta"

# Define the list of beta values to include in the figures
beta_values = [0.05, 0.1, 0.15, 0.25, 0.35, 0.45]

# Dictionary to store merged DataFrames for each beta value
beta_dataframes = {}

# Loop through each beta value
for beta in beta_values:
    beta_str = f"{beta:.3f}"  # Format beta to match filename pattern
    matching_files = glob.glob(os.path.join(folder_path, f"task_{beta_str}*.csv"))

    df_list = [pd.read_csv(file) for file in matching_files]

    if df_list:  # Ensure we have files before concatenating
        merged_df = pd.concat(df_list, ignore_index=True)
        beta_dataframes[beta] = merged_df
    else:
        print(f"No files found for beta = {beta}")

pval_counts = {}
for beta, df in beta_dataframes.items():
    count = (df['pvalue'] < 0.05).sum()
    pval_counts[beta] = count/150
    print(f"Beta {beta:.2f}: {count} p-values > 0.05")

# --- P-value Histogram Figure ---
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # 2 rows, 3 cols

# Define bin edges for P-values (0 to 1 in steps of 0.05)
pvalue_bins = np.arange(0, 1.05, 0.05)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Loop through each beta value and plot P-value histograms
for i, (beta_value, df) in enumerate(beta_dataframes.items()):
    axes[i].hist(df['pvalue'], bins=pvalue_bins, edgecolor='black')
    axes[i].set_title(f'Beta = {beta_value}')
    axes[i].set_xlabel('P-beta')
    axes[i].set_ylabel('Frequency')

# Hide unused subplots if there are any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
pval_counts = {}


# --- n_bad Bar Chart Figure ---
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # 2 rows, 3 cols

# Flatten axes array for easy iteration
axes = axes.flatten()

# Loop through each beta value and plot n_bad bar charts
for i, (beta_value, df) in enumerate(beta_dataframes.items()):
    # Define bin edges for n_bad (increments of 5)
    nbad_bins = np.arange(0, df['nbad'].max() + 10, 5)
    bin_labels = [f"{nbad_bins[j]}-{nbad_bins[j+1]}" for j in range(len(nbad_bins) - 1)]

    # Assign each nbad value to a bin index
    df['nbad_bin'] = np.digitize(df['nbad'], nbad_bins, right=False) - 1
    # Create a full mapping of bins to counts (including empty bins)
    bin_counts = {j: 0 for j in range(len(nbad_bins) - 1)}
    bin_counts.update(df['nbad_bin'].value_counts().to_dict())

    # Convert to lists for plotting
    bin_values = [bin_counts[j] for j in range(len(bin_counts))]

    # Bar chart for n_bad
    axes[i].bar(bin_labels, bin_values, edgecolor='black')
    axes[i].set_title(f'Beta = {beta_value}')
    axes[i].set_xlabel('n_bad')
    axes[i].set_ylabel('Frequency')
    axes[i].set_xticklabels(bin_labels, rotation=45)

# Hide unused subplots if there are any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()