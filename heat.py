from sklearn.model_selection import train_test_split
import simulator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import itertools
import random
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the dataset size
num_points = 500

np.random.seed(1)
tf.keras.utils.set_random_seed(42)
# Generate the dataset
data = simulator.sample_uniform_points(num_points)

X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# new data required for PINN
PDE_points = simulator.sample_uniform_points(1000)
X_PDE_train = PDE_points[:, :-1]

t_bound_1 = np.random.uniform(0, 1, 1000)
X_bound_1_train = np.column_stack([np.ones(1000), t_bound_1])

t_bound_2 = np.random.uniform(0, 1, 1000)
X_bound_2_train = np.column_stack([np.zeros(1000), t_bound_2])

x_init = np.random.uniform(0, 1, 1000)
X_init_train = np.column_stack([x_init, np.zeros(1000)])

# Convert all PDE-related datasets to tensors
X_PDE_train = tf.cast(X_PDE_train, dtype=tf.float32)
X_bound_1_train = tf.cast(X_bound_1_train, dtype=tf.float32)
X_bound_2_train = tf.cast(X_bound_2_train, dtype=tf.float32)
X_init_train = tf.cast(X_init_train, dtype=tf.float32)

epochs = 1000

batch_size = 64
# Create supervised dataset
supervised_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(len(X_train)) \
    .batch(batch_size, drop_remainder=True)

# Create PDE dataset
pde_dataset = tf.data.Dataset.from_tensor_slices(X_PDE_train) \
    .shuffle(len(X_PDE_train)) \
    .batch(batch_size, drop_remainder=True)

# Create boundary dataset
bound_1_dataset = tf.data.Dataset.from_tensor_slices(X_bound_1_train) \
    .shuffle(len(X_bound_1_train)) \
    .batch(batch_size, drop_remainder=True)

bound_2_dataset = tf.data.Dataset.from_tensor_slices(X_bound_2_train) \
    .shuffle(len(X_bound_2_train)) \
    .batch(batch_size, drop_remainder=True)

# Create initial condition dataset
init_dataset = tf.data.Dataset.from_tensor_slices(X_init_train) \
    .shuffle(len(X_init_train)) \
    .batch(batch_size, drop_remainder=True)

# Zip all datasets together so they return batches of the same size
combined_dataset = tf.data.Dataset.zip((supervised_dataset, pde_dataset, bound_1_dataset, bound_2_dataset, init_dataset))


def bound_func(pred, data):
    y_bound = 0
    return pred - y_bound


def init_func(pred, data):
    u_initial = tf.sin(tf.constant(np.pi, dtype=tf.float32) * data[:, 0])  # ✅ Use tf.sin() instead
    u_initial = tf.reshape(u_initial, (-1, 1))
    return (pred - u_initial)

# Define loss function
def pinn_loss(model, X_train, y_train, X_PDE_train, X_bound_1_train, X_bound_2_train, X_init_train, bound_func,
              init_func, alpha, beta, gamma):
    # Supervised Loss
    data_loss = tf.reduce_mean(tf.square(tf.reshape(model(X_train), [-1]) - tf.cast(y_train, tf.float32)))

    # First Gradient Tape for first derivatives
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(X_PDE_train)
        y = model(X_PDE_train)

    # Compute the first derivatives with respect to X_PDE_train
    grads = t1.gradient(y, X_PDE_train)
    y_t = grads[:, 1]  # First derivative with respect to the second column (t)

    # Second Gradient Tape for second derivatives
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(X_PDE_train)
        # Recompute y to get the first derivative again
        y = model(X_PDE_train)
        grads = t2.gradient(y, X_PDE_train)
        y_x = grads[:, 0]  # First derivative with respect to x

    # Compute the second derivative with respect to x
    y_xx = t2.gradient(y_x, X_PDE_train)

    # Extract the second derivative with respect to x (first column of X_PDE_train)
    y_xx = y_xx[:, 0]

    # Compute PDE loss
    pde_loss = tf.reduce_mean(tf.square(y_xx - y_t))

    # Boundary Loss
    bound_loss = (tf.reduce_mean(tf.square(bound_func(model(X_bound_1_train), X_bound_1_train))) +
                  tf.reduce_mean(tf.square(bound_func(model(X_bound_2_train), X_bound_2_train))))

    # Initial cond. loss
    init_loss = tf.reduce_mean(tf.square(init_func(model(X_init_train), X_init_train)))

    # Total Loss
    total_loss = data_loss + alpha * pde_loss + beta*bound_loss + gamma*init_loss
    return total_loss

# Training step function with gradient clipping


X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)
X_combined = tf.cast(X_combined, dtype=tf.float32)
y_combined = tf.cast(y_combined, dtype=tf.float32)
def generate_configs(num_configs=50):
    possible_units = [32, 64, 96, 128]
    configs = []

    # Generate all possible layer structures
    layer_choices = []
    for num_layers in range(1, 4):  # 1 to 3 layers
        layer_choices.extend(itertools.product(possible_units, repeat=num_layers))

    # Ensure each unique layer setup has [0, 0, 0]
    for layers in layer_choices:
        configs.append([list(layers), [0, 0, 0]])

    # Generate additional random configs
    while len(configs) < num_configs:
        layers = list(random.choice(layer_choices))  # Use random.choice, not np.random.choice
        last_three_values = [random.uniform(0, 1) for _ in range(3)]
        configs.append([layers, last_three_values])

    return configs[:num_configs]  # Return only the requested number of configs

configs = generate_configs(10)
for config in configs:
    print(config)

# Dictionary to store test losses for each configuration
validation_losses = {}
# Loop through each layer configuration
for config in configs:
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
    patience = 5
    alpha, beta, gamma = config[1]


    @tf.function
    def train_step(pinn_model, X_batch, y_batch, X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
                   bound_func, init_func, alpha, beta, gamma):
        with tf.GradientTape() as tape:
            loss = pinn_loss(pinn_model, X_batch, y_batch,
                             X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
                             bound_func, init_func, alpha, beta, gamma)  # Adjust alpha as needed
        gradients = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))
        return loss

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0




        for (batch_X, batch_y), batch_X_PDE, batch_X_bound_1, batch_X_bound_2, batch_X_init in combined_dataset:
            # Use custom pinn_loss function here
            batch_loss = train_step(pinn_model, batch_X, batch_y, batch_X_PDE, batch_X_bound_1, batch_X_bound_2, batch_X_init,
                                    bound_func, init_func, alpha, beta, gamma)
            epoch_loss += batch_loss.numpy()
            num_batches += 1

        epoch_loss /= num_batches
        print(f"Epoch {epoch + 1} (Config {config}): Training Loss={epoch_loss}")

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = pinn_model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered for Config {config}. Restoring best weights.")
                pinn_model.set_weights(best_weights)
                break
    val_loss = tf.reduce_mean(tf.square(tf.reshape(pinn_model(X_val), [-1]) - tf.cast(y_val, tf.float32))).numpy()
    validation_losses[tuple(tuple(sublist) for sublist in config)] = val_loss
    print(f"Validation Loss for Config {config}: {val_loss}")

    # Evaluate the PINN model
    test_loss_pinn = tf.reduce_mean(
        tf.square(tf.reshape(pinn_model(X_test), [-1]) - tf.cast(y_test, tf.float32))).numpy()
    print(f"Test Loss for pinn {config}: {test_loss_pinn}")

# Print out the test losses for all configurations
print("val Losses for all configurations:", validation_losses)

best_config = min(validation_losses, key=validation_losses.get)
min_val_loss = validation_losses[best_config]

print(f"Best Configuration: {best_config} with Validation Loss: {min_val_loss}")


full_supervised_dataset = tf.data.Dataset.from_tensor_slices((X_combined, y_combined)) \
    .shuffle(len(X_combined)) \
    .batch(batch_size, drop_remainder=True)

# Zip all datasets together so they return batches of the same size
full_combined_dataset = tf.data.Dataset.zip((full_supervised_dataset, pde_dataset, bound_1_dataset, bound_2_dataset, init_dataset))



# Build the PINN model
pinn_model = Sequential()
pinn_model.add(Dense(best_config[0][0], activation='tanh'))  # First layer with input dimension
for units in best_config[0][1:]:
    pinn_model.add(Dense(units, activation='tanh'))  # Add hidden layers
pinn_model.add(Dense(1))  # Output layer

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # You can adjust the learning rate

# Early stopping variables
best_loss = float('inf')
best_weights = pinn_model.get_weights()
wait = 0
patience = 5
alpha, beta, gamma = best_config[1]

@tf.function
def train_step(pinn_model, X_batch, y_batch, X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
               bound_func, init_func, alpha, beta, gamma):
    with tf.GradientTape() as tape:
        loss = pinn_loss(pinn_model, X_batch, y_batch,
                         X_PDE_batch, X_bound_1_batch, X_bound_2_batch, X_init_batch,
                         bound_func, init_func, alpha, beta, gamma)  # Adjust alpha as needed
    gradients = tape.gradient(loss, pinn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0

    for (batch_X, batch_y), batch_X_PDE, batch_X_bound_1, batch_X_bound_2, batch_X_init in full_combined_dataset:
        # Use custom pinn_loss function here
        batch_loss = train_step(pinn_model, batch_X, batch_y, batch_X_PDE, batch_X_bound_1, batch_X_bound_2, batch_X_init,
                                bound_func, init_func, alpha, beta, gamma)
        epoch_loss += batch_loss.numpy()
        num_batches += 1

    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1} (Config {best_config}): Training Loss={epoch_loss}")

    # Early stopping logic
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_weights = pinn_model.get_weights()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered for Config {best_config}. Restoring best weights.")
            pinn_model.set_weights(best_weights)
            break
# Evaluate the PINN model
test_loss_pinn = tf.reduce_mean(
    tf.square(tf.reshape(pinn_model(X_test), [-1]) - tf.cast(y_test, tf.float32))).numpy()
print(f"Test Loss for pinn {best_config}: {test_loss_pinn}")