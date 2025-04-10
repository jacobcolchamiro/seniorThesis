import glob
import os
import pandas as pd

csv_files = glob.glob(os.path.join('output_pinn2', "*.csv"))

# Store (config, val_loss) tuples
config_losses = []

for file in csv_files:
    df = pd.read_csv(file)
    config = df["config"][0]  # Assuming one row per file
    val_loss = df["val_loss"][0]

    config_losses.append((config, val_loss))

# Sort by validation loss in ascending order
config_losses.sort(key=lambda x: x[1])

# Get the top 3 configurations
top_3_configs = config_losses

for rank, (config, val_loss) in enumerate(top_3_configs, start=1):
    print(f"Rank {rank}: Config = {config}, Val Loss = {val_loss}")

print(top_3_configs)