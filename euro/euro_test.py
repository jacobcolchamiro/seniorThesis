from sklearn.model_selection import train_test_split
import euro_model_train
import euro_simulator
import numpy as np
import pandas as pd
from scipy.stats import binom
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import itertools

# Fixed architecture
architecture = [96, 96, 96]

# Possible weight values (expanded)
pde_weights = [1e-4, 1e-3, 1e-2, 1e-1, 1]
boundary_weights = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
initial_weights = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Generate all combinations
configs = [
    [architecture, list(weights)]
    for weights in itertools.product(pde_weights, boundary_weights, initial_weights)
]




seed = 42
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

df = pd.read_csv('euro/option_data.csv')
df = df.dropna()
df = df.sample(n=15000, random_state=seed)
y = np.array(df['price'])
X = df.drop(columns='price').to_numpy()
X_scaler = StandardScaler().fit(X)
X_transform = X_scaler.transform(X)
# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_transform, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# **Feature Scaling (Fit only on Training Data)**
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
X_PDE = np.empty_like(X_train)

# For each feature, randomly sample it independently from the training data
for i in range(X_train.shape[1]):  # Loop through each feature (column)
    X_PDE[:, i] = np.random.choice(X_train[:, i], size=len(X_train), replace=True)

sec_bound_1 = -means[3]/stds[3]
X_bound_1 = X_PDE.copy()
X_bound_1[:, 3] = sec_bound_1# Assuming S is the 1st column
#X_bound_1 = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, [sec_bound_1, sec_bound_1], r_range)

sec_bound_2 = (X[:, 3].max()*2-means[3])/stds[3]
X_bound_2 = X_PDE.copy()
X_bound_2[:, 3] = sec_bound_2
#X_bound_2 = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, [sec_bound_2, sec_bound_2], r_range)

t_init = 0
X_init = X_PDE.copy()
X_init[:, 2] = t_init
#X_init = euro_simulator.sample_features(len(X), K_range, sig_range, [t_init, t_init], sec_range, r_range)
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
#idx = 0
configs = [configs[idx]]
#
# x_init = np.random.uniform(0, 1, 1000)
# X_init = np.column_stack([x_init, np.zeros(1000)])
#ff_nn = euro_model_train.train_pinn(False, X_train, y_train, X_val, y_val,
#                                X_PDE, X_bound_1, X_bound_2, X_init, 128, means, stds, configs,
#                               seed=seed)




#df.to_csv(f'output_{configs[0]}.csv', index =False)
print(configs)
pinn = euro_model_train.train_pinn(True, X_train, y_train, X_val, y_val,
                                 X_PDE, X_bound_1, X_bound_2, X_init, 128, means, stds, configs,
                                seed=seed)
#print(pinn)
df = pd.DataFrame([{"config": configs[0], "val_loss": pinn}])
df.to_csv(f"output_pinn/task_{configs[0]}.csv", index=False)

# conformal_width = conformal.split_conformal_width(ff_nn, X_test, y_test, beta=0.25)
# new_test_data = euro_simulator.sample_features(len(X), K_range, sig_range, tte_range, sec_range, r_range)
# new_test_X = new_test_data
# #
# ff_nn_pred = ff_nn(X_test)
# ff_nn_pred = ff_nn(new_test_X)
# pinn_pred = pinn(new_test_X)
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
# prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), 0.25)
# print(prob)
# print(n_bad)









