from sklearn.model_selection import train_test_split
import model_train
import simulator
import conformal
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker



np.random.seed(1)
p_values = {}
epsilon = [0, 0.05, 0.1, 0.5]
alpha = 0.25

for eps in epsilon:
    data = simulator.sample_uniform_points(256, eps = eps)
    X = data[:, :-1]  # Features
    y = data[:, -1]  # Labels

    # Split dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # New data required for PINN
    PDE_points = simulator.sample_uniform_points(1000, eps = eps)
    X_PDE = PDE_points[:, :-1]

    t_bound_1 = np.random.uniform(0, 1, 1000)
    X_bound_1 = np.column_stack([np.ones(1000), t_bound_1])

    t_bound_2 = np.random.uniform(0, 1, 1000)
    X_bound_2 = np.column_stack([np.zeros(1000), t_bound_2])

    x_init = np.random.uniform(0, 1, 1000)
    X_init = np.column_stack([x_init, np.zeros(1000)])

    ff_nn = model_train.train_pinn(False, X_train, y_train, X_val, y_val, X_PDE, X_bound_1, X_bound_2, X_init,
                                   batch_size=64, bound = 0)
    pinn = model_train.train_pinn(True, X_train, y_train, X_val, y_val, X_PDE, X_bound_1, X_bound_2, X_init,
                                  batch_size=64, bound = 0)

    conformal_width = conformal.split_conformal_width(ff_nn, X_test, y_test, alpha=alpha)
    new_test_data = simulator.sample_uniform_points((int)(256 * 0.15), eps = eps)
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
    prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), alpha)
    print(f"sample size: {eps}, P(X >= {n_bad}) = {prob}")
    p_values[eps] = prob

p_values_x = list(p_values.keys())
p_values_y = list(p_values.values())

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

sns.lineplot(x=p_values_x, y=p_values_y, marker='o', linestyle='-', color='b')
plt.xlabel("Noise standard deviation")
plt.ylabel("p-beta")
plt.title("p-beta vs standard deviation of noise")
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.1))
plt.tight_layout()
plt.show()






