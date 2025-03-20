from sklearn.model_selection import train_test_split
import model_train
import simulator
import conformal
import numpy as np
import pandas as pd
import os
from scipy.stats import binom
import tensorflow as tf


idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
np.random.seed(1)
betas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.495]
beta = betas[idx // 150]
bob = idx % 150
# results = {beta: {'pvalues': [], 'nbad': []} for beta in betas}

pvalues_ = list()
nbad_ = list()


seed = bob
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

data = simulator.sample_uniform_points(300, eps = 0.05)
X = data[:, :-1]  # Features
y = data[:, -1]  # Labels

# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# New data required for PINN
PDE_points = simulator.sample_uniform_points(1000, eps = 0.05)
X_PDE = PDE_points[:, :-1]

t_bound_1 = np.random.uniform(0, 1, 1000)
X_bound_1 = np.column_stack([np.ones(1000), t_bound_1])

t_bound_2 = np.random.uniform(0, 1, 1000)
X_bound_2 = np.column_stack([np.zeros(1000), t_bound_2])

x_init = np.random.uniform(0, 1, 1000)
X_init = np.column_stack([x_init, np.zeros(1000)])

ff_nn = model_train.train_pinn(False, X_train, y_train, X_val, y_val, X_PDE, X_bound_1, X_bound_2, X_init,
                               batch_size=64, bound = 0, seed=seed)
pinn = model_train.train_pinn(True, X_train, y_train, X_val, y_val, X_PDE, X_bound_1, X_bound_2, X_init,
                              batch_size=64, bound = 0, seed = seed)

conformal_width = conformal.split_conformal_width(ff_nn, X_test, y_test, beta=beta)
new_test_data = simulator.sample_uniform_points((int)(300 * 0.15), eps = 0.05)
new_test_X = new_test_data[:, :-1]

ff_nn_pred = ff_nn(new_test_X)
pinn_pred = pinn(new_test_X)
lower = ff_nn_pred - conformal_width
upper = ff_nn_pred + conformal_width

min_diffs = np.zeros(len(lower))
max_diffs = np.zeros(len(upper))

for i in range(len(lower)):
    y_vals = np.linspace(lower[i], upper[i], num=1000)
    diff_residuals = np.reshape(np.abs(ff_nn_pred[i] - y_vals), -1) - np.reshape(np.abs(pinn_pred[i] - y_vals), -1)
    min_diffs[i] = np.min(diff_residuals)
    max_diffs[i] = np.max(diff_residuals)

n_bad = len(max_diffs[max_diffs < 0])
prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), beta)
# results[beta]['pvalues'].append(prob)
# results[beta]['nbad'].append(n_bad)
pvalues_.append(prob)
nbad_.append(n_bad)





# np.savetxt(f"output/task_{idx}.csv", results, delimiter=",")
df = pd.DataFrame({"pvalue": pvalues_, "nbad": nbad_})
df.to_csv(f"output/task_{beta:.04f}_{bob:03d}.csv", index=False)







