import tensorflow as tf
import itertools
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


NUM_CONFIGS = 75

def train_pinn(pinn, X_train, y_train, X_val, y_val, X_PDE, X_bound_1, X_bound_2, X_init, batch_size, means, stds, configs, epochs=30000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
    # if pinn:
    #     configs = generate_configs(True, NUM_CONFIGS)
    # else:
    #     configs = generate_configs(False, NUM_CONFIGS)

    # # Cast data to float32 for compatibility with TensorFlow
    #
    X_PDE = tf.cast(X_PDE, dtype=tf.float32)
    X_bound_1 = tf.cast(X_bound_1, dtype=tf.float32)
    X_bound_2 = tf.cast(X_bound_2, dtype=tf.float32)
    X_init = tf.cast(X_init, dtype=tf.float32)

    # Create datasets for each data type
    supervised_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
        .shuffle(len(X_train)) \
        .batch(batch_size, drop_remainder=True)

    pde_dataset = tf.data.Dataset.from_tensor_slices(X_PDE) \
        .shuffle(len(X_PDE)) \
        .batch(batch_size, drop_remainder=True)

    bound_1_dataset = tf.data.Dataset.from_tensor_slices(X_bound_1) \
        .shuffle(len(X_bound_1)) \
        .batch(batch_size, drop_remainder=True)

    bound_2_dataset = tf.data.Dataset.from_tensor_slices(X_bound_2) \
        .shuffle(len(X_bound_2)) \
        .batch(batch_size, drop_remainder=True)

    init_dataset = tf.data.Dataset.from_tensor_slices(X_init) \
        .shuffle(len(X_init)) \
        .batch(batch_size, drop_remainder=True)

    # Compute batch counts
    batch_counts = {
        "supervised": tf.data.experimental.cardinality(supervised_dataset).numpy()
        ,
        "pde": tf.data.experimental.cardinality(pde_dataset).numpy(),
        "bound_1": tf.data.experimental.cardinality(bound_1_dataset).numpy(),
        "bound_2": tf.data.experimental.cardinality(bound_2_dataset).numpy(),
        "init": tf.data.experimental.cardinality(init_dataset).numpy()
    }


    # Find the largest batch size
    max_batches = max(batch_counts.values())

    # Repeat the smallest dataset(s) to match the largest batch size
    if batch_counts["supervised"] < max_batches:
        supervised_dataset = supervised_dataset.repeat().take(max_batches)

    # Zip all datasets together for consistent batching
    combined_dataset = tf.data.Dataset.zip(
        (supervised_dataset
         , pde_dataset, bound_1_dataset, bound_2_dataset, init_dataset
         ))

    # Get the best model configuration based on validation data
    best_structure = choose_config(combined_dataset, X_val, y_val, configs, epochs, means, stds)

    # # Combine the training and validation data
    # X_combined = np.concatenate([X_train, X_val], axis=0)
    # y_combined = np.concatenate([y_train, y_val], axis=0)
    #
    # # Create the final supervised dataset with combined data
    # X_combined = tf.cast(X_combined, dtype=tf.float32)
    # y_combined = tf.cast(y_combined, dtype=tf.float32)
    #
    # supervised_dataset = tf.data.Dataset.from_tensor_slices((X_combined, y_combined)) \
    #     .shuffle(len(X_combined)) \
    #     .batch(batch_size, drop_remainder=True)
    #
    # # Re-zip the combined dataset for training
    # combined_dataset = tf.data.Dataset.zip(
    #     (supervised_dataset, pde_dataset, bound_1_dataset, bound_2_dataset, init_dataset))
    #
    # # Train the best model
    # best_model = get_model(combined_dataset, best_structure, epochs, means, stds)
    return best_structure[1]


def generate_configs(pinn, num_configs=NUM_CONFIGS):
    possible_units = [32, 64, 96, 128]
    configs = []

    # Generate all possible layer structures
    layer_choices = []
    for num_layers in range(2, 5):  # 2 to 4 layers
        layer_choices.extend(itertools.product(possible_units, repeat=num_layers))

    # Ensure each unique layer setup has [0, 0, 0]
    if not pinn:
        for layers in layer_choices:
            configs.append([list(layers), [0, 0, 0]])

    if pinn:
        while len(configs) < num_configs:
            layers = list(random.choice(layer_choices))  # Use random.choice, not np.random.choice
            last_three_values = [random.uniform(0.5, 1) for _ in range(3)]
            configs.append([layers, last_three_values])
    random.shuffle(configs)
    num_configs = min(num_configs, len(configs))
    return configs[:num_configs]  # Return only the requested number of configs

def bound_func_1(pred):
    y_bound = 0
    return pred - y_bound

def bound_func_2(pred, data, means, stds):
    bound = data[:, 3][0]
    y_bound = bound*stds[3]+means[3] - (data[:, 0]*stds[0] + means[0])*tf.exp(-(data[:, 4]*stds[4]+means[4])*(data[:, 2]*stds[2]+means[2]))
    y_bound = tf.reshape(y_bound, (-1, 1))
    return pred - y_bound


def init_func(pred, data, means, stds):
    y_initial = tf.maximum((data[:, 3]*stds[3] + means[3]) - (data[:, 0]*stds[0]+means[0]), 0)
    y_initial = tf.reshape(y_initial, (-1, 1))
    return (pred - y_initial)

def choose_config(combined_dataset, X_val, y_val, configs, epochs, means, stds):


    # Dictionary to store test losses for each configuration
    validation_losses = {}
    # Loop through each layer configuration
    for config in configs:
        print(f'starting config: {config}')
        pinn_model = get_model(combined_dataset, config, epochs, means, stds)
        val_loss = tf.reduce_mean(tf.square(tf.reshape(pinn_model(X_val), [-1]) - tf.cast(y_val, tf.float32))).numpy()
        print(f'config: {config}, val loss: {val_loss}')
        validation_losses[tuple(tuple(sublist) for sublist in config)] = val_loss
    best_config = min(validation_losses, key=validation_losses.get)
    min_val_loss = validation_losses[best_config]  # Get the corresponding minimum loss

    return best_config, min_val_loss

def get_model(combined_dataset, config, epochs, means, stds):
    # Build the PINN model
    pinn_model = Sequential()
    pinn_model.add(Dense(config[0][0], activation='tanh'))  # First layer with input dimension
    for units in config[0][1:]:
        pinn_model.add(Dense(units, activation='tanh'))  # Add hidden layers
    pinn_model.add(Dense(1))  # Output layer

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # You can adjust the learning rate

    # Early stopping variables
    best_loss = float('inf')
    best_weights = pinn_model.get_weights()
    wait = 0
    patience = 25
    alpha, beta, gamma = config[1]

    @tf.function
    def train_step(pinn_model, X_batch, y_batch
                   , X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
                   bound_func_1, bound_func_2, init_func, alpha, beta, gamma, means, stds
    ):
        try:
            with tf.GradientTape() as tape:
                loss = pinn_loss(pinn_model, X_batch, y_batch,
                                  X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
                                  bound_func_1, bound_func_2, init_func, alpha, beta, gamma, means, stds
                                 )  # Adjust alpha as needed
            gradients = tape.gradient(loss, pinn_model.trainable_variables)

            # Use tf.reduce_any() instead of Python 'if' for TensorFlow compatibility
            has_invalid_gradients = tf.reduce_any(
                [tf.reduce_all(tf.equal(g, 0)) if g is not None else True for g in gradients])

            if has_invalid_gradients:
                return loss  # Return current loss without updating

            optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))
            return loss

        except tf.errors.InvalidArgumentError as e:
            return tf.constant(0.0)  # Return a dummy loss to keep training running

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for ((batch_X, batch_y), batch_X_PDE, batch_X_bound_1, batch_X_bound_2, batch_X_init) in combined_dataset:
             # :
            # Use custom pinn_loss function here
            batch_loss = train_step(pinn_model, batch_X, batch_y
                                    , batch_X_PDE, batch_X_bound_1, batch_X_bound_2,
                                     batch_X_init,
                                     bound_func_1, bound_func_2, init_func, alpha, beta, gamma, means, stds
             )
            epoch_loss += batch_loss.numpy()
            num_batches += 1

        epoch_loss /= num_batches
        #print(f'epoch {epoch}: loss {epoch_loss}')
        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = pinn_model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                pinn_model.set_weights(best_weights)
                break

    return pinn_model
def pinn_loss(model, X, y, X_PDE, X_bound_1, X_bound_2, X_init, bound_func_1, bound_func_2,init_func, alpha, beta, gamma, means, stds):
    data_loss = tf.reduce_mean(tf.square(tf.reshape(model(X), [-1]) - tf.cast(y, tf.float32)))
    # First Gradient Tape for first derivatives
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(X_PDE)
        y = model(X_PDE)
    #
    # Compute the first derivative wrt tte
    grads = t1.gradient(y, X_PDE)
    y_t = -1*grads[:, 2]
    y_s = grads[:, 3]
    # Second Gradient Tape for second derivatives
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(X_PDE)
        # Recompute y to get the first derivative again
        y = model(X_PDE)
        grads = t2.gradient(y, X_PDE)
        y_s = grads[:, 3]  # First derivative with respect to x

    # Compute the second derivative with respect to x
    y_ss = t2.gradient(y_s, X_PDE)

    # # Extract the second derivative with respect to x (maybe first column of X_PDE_train)
    y_ss = y_ss[:, 3]

    # Ensure stds and means are float32
    stds = tf.cast(stds, dtype=tf.float32)
    means = tf.cast(means, dtype=tf.float32)

    # Ensure X_PDE is float32
    X_PDE = tf.cast(X_PDE, dtype=tf.float32)

    # Ensure all computed terms are float32
    y_t = tf.cast(y_t, dtype=tf.float32)
    y_ss = tf.cast(y_ss, dtype=tf.float32)
    y_s = tf.cast(y_s, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    # # Compute PDE loss
    pde_loss = tf.reduce_mean(tf.square(-(1/stds[2])*y_t + 0.5*tf.square(stds[1]*X_PDE[:, 1]+means[1])*tf.square(stds[3]*X_PDE[:, 3]+means[3])*(1/stds[3]**2)*y_ss + (stds[4]*X_PDE[:, 4]+means[4])*(stds[3]*X_PDE[:, 3]+means[3])*(1/stds[3])*y_s-(stds[4]*X_PDE[:, 4]+means[4])*tf.reshape(y, (-1,))))

    # # Boundary Loss
    bound_loss = (tf.reduce_mean(tf.square(bound_func_1(model(X_bound_1)))) + tf.reduce_mean(tf.square(bound_func_2(model(X_bound_2), X_bound_2, means, stds))))

    # # Initial cond. loss
    init_loss = tf.reduce_mean(tf.square(init_func(model(X_init), X_init, means, stds)))

    # # Total Loss
    # total_loss = data_loss + alpha * pde_loss + beta * bound_loss + gamma * init_loss
    total_loss = data_loss + alpha*pde_loss + beta*bound_loss + gamma*init_loss
    return total_loss