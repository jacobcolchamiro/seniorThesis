from sklearn.model_selection import train_test_split
import euro_model_train
import euro_simulator
import euro_conformal
import numpy as np
import pandas as pd
from scipy.stats import binom
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



# Fixed architecture
nn_config = [[96, 96, 96], [0, 0, 0]]
pinn_config = [[96, 96, 96], [0.0001, 0.1, 0.0001]]

seed = 42
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

df = pd.read_csv('option_data.csv')
df = df.dropna()
df = df.sample(n=15000, random_state=seed)
y = np.array(df['price'])
X = df.drop(columns='price').to_numpy()
X_scaler = StandardScaler().fit(X)
X_transform = X_scaler.transform(X)
# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_transform, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


  # Standardize inputs using only training data
means = X_scaler.mean_
stds = X_scaler.scale_


# New data required for PINN
K_range = (X_transform[:, 0].min(), X_transform[:, 0].max())
sig_range = (X_transform[:, 1].min(), X_transform[:, 1].max())
tte_range = (X_transform[:, 2].min(), X_transform[:, 2].max())
sec_range = (X_transform[:, 3].min(), X_transform[:, 3].max())
r_range = (X_transform[:, 4].min(), X_transform[:, 4].max())

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

t_init = 0
X_init = X_PDE.copy()
X_init[:, 2] = t_init

ff_nn = load_model("models/ffnn_model.h5")
pinn = load_model("models/pinn_model.h5")

conformal_width = euro_conformal.split_conformal_width(ff_nn, X_test, y_test, beta=0.05)

# Sample extra points
X_extra_df = df.sample(n=300, random_state=42)
X_extra = X_extra_df.drop(columns='price').to_numpy()

# Ensure X_extra contains no duplicate points from X
X_set = {tuple(row) for row in X}  # Convert X to a set of tuples for quick lookup

for i in range(len(X_extra)):
    while tuple(X_extra[i]) in X_set:
        new_sample = df.sample(n=1, random_state=None).drop(columns='price').to_numpy()[0]
        X_extra[i] = new_sample  # Replace duplicate with a new sample

# Standardize X_extra using the same scaler
X_extra_standardized = X_scaler.transform(X_extra)

# For each feature, randomly sample it independently from the training data
for i in range(X_train.shape[1]):  # Loop through each feature (column)
    new_test_X[:, i] = np.random.choice(X_train[:, i], size=len(X_train), replace=True)





ff_nn_pred = ff_nn(new_test_X)
pinn_pred = pinn(new_test_X)
lower = ff_nn_pred - conformal_width
upper = ff_nn_pred + conformal_width
# #
min_diffs = np.zeros(len(lower))
max_diffs = np.zeros(len(upper))
# #
for i in range(len(lower)):
    y_vals = np.linspace(lower[i], upper[i], num=1000)
    diff_residuals = np.reshape(np.abs(ff_nn_pred[i] - y_vals), -1) - np.reshape(np.abs(pinn_pred[i] - y_vals), -1)
    min_diffs[i] = np.min(diff_residuals)
    max_diffs[i] = np.max(diff_residuals)
# #
n_bad = len(max_diffs[max_diffs < 0])
prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), 0.05)
print(prob)
print(n_bad)

