import numpy as np


# generate heat equation dataset
def sample_features(num_points, strike_range, vol_range, tte_range, sec_range, r_range):
    K = np.random.uniform(strike_range[0], strike_range[1], num_points)
    sig = np.random.uniform(vol_range[0], vol_range[1], num_points)
    tte = np.random.uniform(tte_range[0], tte_range[1], num_points)
    sec = np.random.uniform(sec_range[0], sec_range[1], num_points)
    r = np.random.uniform(r_range[0], r_range[1], num_points)

    return np.column_stack([K, sig, tte, sec, r])