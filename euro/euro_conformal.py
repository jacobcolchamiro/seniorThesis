import tensorflow as tf
import numpy as np



def split_conformal_width(model, X_test, y_test, beta):
    pred = tf.reshape(model(X_test), [-1])
    abs_resid = tf.abs(pred-y_test)
    sorted_resid = tf.sort(abs_resid)

    k = int(np.ceil((len(y_test)+ 1) * (1 - beta)))

    # Ensure k is within valid bounds
    k = min(max(k, 1), len(sorted_resid))

    # Get the k-th smallest residual
    d = sorted_resid[k - 1].numpy()  # Convert to NumPy scalar for easier handling
    return d
