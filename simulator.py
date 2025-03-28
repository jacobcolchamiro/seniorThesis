
import numpy as np


# generate heat equation dataset
def sample_uniform_points(num_points, x_range=(0, 1), t_range=(0, 1), eps = 0):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    t = np.random.uniform(t_range[0], t_range[1], num_points)
    y = np.exp(-1 * (np.pi**2) * t) * np.sin(np.pi*x)
    # Add random noise to y
    noise = np.random.normal(0, eps, len(y))
    y_noisy = y + noise
    #y_noisy = y

    return np.column_stack([x, t, y_noisy])


