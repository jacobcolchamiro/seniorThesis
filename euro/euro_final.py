from sklearn.model_selection import train_test_split
import euro_model_train
import euro_simulator
import numpy as np
import pandas as pd
from scipy.stats import binom
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import itertools
import euro_conformal

BETA = 0.25
pinn_configs = [[[32, 32, 128], [0.0001, 0.001, 0.0001]],
[[32, 32, 128], [0.0001, 0.001, 0.001]],
[[32, 32, 128], [0.0001, 0.01, 0.0001]]]


nn_configs = [[[64, 128], [0, 0, 0]],
              [[96, 128], [0, 0, 0]],
              [[32, 32, 128], [0, 0, 0]],
              [[96, 96, 128], [0, 0, 0]],
              [[64, 96, 128], [0, 0, 0]],
              [[96, 64, 128], [0, 0, 0]],
              [[32, 128, 128], [0, 0, 0]],
              [[32, 96, 96, 128], [0, 0, 0]],
              [[96, 64, 128, 128], [0, 0, 0]],
              [[64, 64, 128, 128], [0, 0, 0]],
              [[32, 64, 64, 128], [0, 0, 0]],
              [[32, 32, 96, 128], [0, 0, 0]],
              [[32, 128, 96, 128], [0, 0, 0]]]


seed = 42
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

df = pd.read_csv('euro/option_data.csv').dropna()

# Get all indices
all_indices = np.arange(len(df))

# 1. Sample 15,000 indices for your main dataset X
main_size = 15000
main_indices = np.random.choice(all_indices, size=main_size, replace=False)

# Create the main dataset X and y from df using these indices
X = df.iloc[main_indices].drop(columns='price').to_numpy()
y = df.iloc[main_indices]['price'].to_numpy()

# 2. Remove the main_indices from the full set and sample extra indices for X_extra
remaining_indices = np.setdiff1d(all_indices, main_indices)
extra_size = 3000  # or however many extra points you want
extra_indices = np.random.choice(remaining_indices, size=extra_size, replace=False)

# Create the extra dataset for the conformal hypothesis test
X_extra = df.iloc[extra_indices].drop(columns='price').to_numpy()
y_extra = df.iloc[extra_indices]['price'].to_numpy()

# 3. Standardize the main dataset X using StandardScaler
X_scaler = StandardScaler().fit(X)
X_transform = X_scaler.transform(X)

# Standardize X_extra using the same scaler
X_extra_transform = X_scaler.transform(X_extra)

# 4. Split the main dataset into training, validation, and test sets.
# (For example, 70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_transform, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)



  # Standardize inputs using only training data
means = X_scaler.mean_
stds = X_scaler.scale_


# New data required for PINN
# K_range = (X_transform[:, 0].min(), X_transform[:, 0].max())
# sig_range = (X_transform[:, 1].min(), X_transform[:, 1].max())
# tte_range = (X_transform[:, 2].min(), X_transform[:, 2].max())
# sec_range = (X_transform[:, 3].min(), X_transform[:, 3].max())
# r_range = (X_transform[:, 4].min(), X_transform[:, 4].max())

#X_PDE = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, sec_range, r_range)
X_PDE = X_train[np.random.choice(len(X_train), size=len(X_train), replace=True)]


sec_bound_1 = -means[3]/stds[3]
X_bound_1 = X_train[np.random.choice(len(X_train), size=len(X_train), replace=True)]
X_bound_1[:, 3] = sec_bound_1# Assuming S is the 1st column
#X_bound_1 = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, [sec_bound_1, sec_bound_1], r_range)

sec_bound_2 = (X[:, 3].max()*2-means[3])/stds[3]
X_bound_2 = X_train[np.random.choice(len(X_train), size=len(X_train), replace=True)]

X_bound_2[:, 3] = sec_bound_2

t_init = 0
X_init = X_train[np.random.choice(len(X_train), size=len(X_train), replace=True)]
X_init[:, 2] = t_init

# ff_nn = load_model("ffnn.h5")
# pinn = load_model("pinn.h5")
# Fixed architecture



idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

configs = [nn_configs[idx]]

ffnn = euro_model_train.train_pinn(False, X_train, y_train, X_val, y_val,
                              X_PDE, X_bound_1, X_bound_2, X_init, 128, means, stds, configs,
                             seed=seed)
X_joint = np.concatenate((X_train, X_val), axis=0)

y_joint = np.concatenate((y_train, y_val), axis=0)
final_loss = tf.reduce_mean(tf.square(tf.reshape(ffnn(X_joint), [-1]) - tf.cast(y_joint, tf.float32)))
ffnn.save(f"saved_models3/pinn_model_{configs[0]}.h5")

df = pd.DataFrame([{"config": configs[0], "final_loss": final_loss}])
df.to_csv(f"saved_models3/loss_{configs[0]}.csv", index=False)
#

# Extract training loss from ffnn[1]
#loss_history = ffnn[1]

# Save trained model
#ffnn[0].save("saved_models/ffnn_model.h5")


# plt.figure(figsize=(8, 5))
# plt.plot(ffnn[1], linestyle="-", color="b", label="Training Loss")
# plt.xlabel("Training Time (s)")
# plt.ylabel("Training Loss")
# plt.title("Epoch Loss vs. Training Time")
# plt.legend()
# plt.grid()
# #plt.savefig("output_pinn/loss_curve.png")
# plt.show()
#
# # Save test results
# results_df = pd.DataFrame([{"config": nn_config, "training_Loss": loss_history}])
# results_df.to_csv("ffnn_results.csv", index=False)
#
# # Load model for evaluation
# ffnn_model = load_model("saved_models/ffnn_model.h5")




#ffnn = load_model("models/ffnn_model.h5")






#
#
# pinn = euro_model_train.train_pinn(True, X_train, y_train, X_val, y_val,
#                                  X_PDE, X_bound_1, X_bound_2, X_init, 128, means, stds, [pinn_config],
#                                 seed=seed)
# pinn.save("saved_models/pinn_model.h5")
#print(pinn)
# df = pd.DataFrame([{"config": configs[0], "val_loss": pinn}])
# df.to_csv(f"output_pinn/task_{configs[0]}.csv", index=False)

# conformal_width = euro_conformal.split_conformal_width(ff_nn, X_test, y_test, BETA)
# # new_test_data = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, sec_range, r_range)
# # new_test_X = new_test_data
# #
# ff_nn_pred = ff_nn(X_extra_transform)
# pinn_pred = pinn(X_extra_transform)
# lower = ff_nn_pred - conformal_width
# upper = ff_nn_pred + conformal_width
# #
# min_diffs = np.zeros(len(lower))
# max_diffs = np.zeros(len(upper))
# #
# for i in range(len(lower)):
#     y_vals = np.linspace(lower[i], upper[i], num=1000)
#     diff_residuals = np.reshape(np.abs(ff_nn_pred[i] - y_vals), -1) - np.reshape(np.abs(pinn_pred[i] - y_vals), -1)
#     min_diffs[i] = np.min(diff_residuals)
#     max_diffs[i] = np.max(diff_residuals)
# #
# n_bad = len(max_diffs[max_diffs < 0])
# prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), BETA)
# print(prob)
# print(n_bad)





