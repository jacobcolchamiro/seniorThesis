from sklearn.model_selection import train_test_split
import simulator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Define the dataset size
num_points = 500

np.random.seed(1)
# Generate the dataset (assuming you've already used the sample_uniform_points function)
data = simulator.sample_uniform_points(num_points)
# Assuming `data` is a 2D array with features + labels, split it
X = data[:, :-1]  # All columns except the last one (features)
y = data[:, -1]   # Last column (labels)

# First split: Training and temporary set (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training

# Second split: Validation and test sets from the temporary set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15% test


def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 1, 3)):  # Tune 1 to 3 layers
        model.add(Dense(
            units=hp.Int('units', 32, 128, step=32),  # Tune units
            activation=hp.Choice('activation', ['tanh'])  # Tune activation
        ))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse')
    # set random seed to specific number -> for all nets that are trained
    return model



# Create the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2
)

# Perform the search
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Get the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print('Best hyperparameter choices for Feed Forward Neural Network: ')
for key, value in best_hps.values.items():
    print(f"{key}: {value}")

# Train the best model
X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

final_model = tuner.hypermodel.build(best_hps)

# Train the final model, switched to do early stopping with rain loss rather than validation
final_model.fit(
    X_combined, y_combined,
    epochs=100,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
)

# Evaluate the model on the test set
test_loss_nn = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss_nn}")




# retrieve feature points across domain for PINN training
# PDE_points = simulator.sample_uniform_points(1000)
# X_PDE = PDE_points[:, :-1]
# X_PDE_train, X_PDE_val = train_test_split(X_PDE, test_size=0.2, random_state=42)
#
# t_bound_1 = np.random.uniform(0, 1, 1000)
# X_bound_1 = np.column_stack([np.ones(1000), t_bound_1])
# X_bound_1_train, X_bound_1_val = train_test_split(X_bound_1, test_size=0.2, random_state=42)
#
# t_bound_2 = np.random.uniform(0, 1, 1000)
# X_bound_2 = np.column_stack([np.zeros(1000), t_bound_2])
# X_bound_2_train, X_bound_2_val = train_test_split(X_bound_2, test_size=0.2, random_state=42)
#
# x_init = np.random.uniform(0, 1, 1000)
# X_init = np.column_stack([x_init, np.zeros(1000)])
# X_init_train, X_init_val = train_test_split(X_init, test_size=0.2, random_state=42)
#


# Build the PINN using the selected hyperparameters
new_model = tuner.hypermodel.build(best_hps)


def pinn_loss(model, X_train, y_train, X_PDE_train, X_bound_1_train, X_bound_2_train, X_init_train, bound_func,
              init_func, alpha):
    # Supervised Loss
    # data_loss = tf.reduce_mean(tf.square(tf.reshape(model(X_train), [-1]) - tf.cast(y_train, tf.float32)))

    return tf.reduce_mean(tf.square(tf.reshape(model(X_train), [-1]) - tf.cast(y_train, tf.float32)))

    # Convert X_PDE_train to a tensor and ensure it's watched
    X_PDE_train = tf.cast(X_PDE_train, dtype=tf.float32)

    # First Gradient Tape for first derivatives
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(X_PDE_train)
        y = model(X_PDE_train)

    # Compute the first derivatives with respect to X_PDE_train
    grads = t1.gradient(y, X_PDE_train)
    y_x = grads[:, 0]  # First derivative with respect to the first column (x)
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

    # Check if second derivative is None
    if y_xx is None:
        raise ValueError("Second derivative y_xx computation failed. Ensure X_PDE_train is properly watched.")

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
    total_loss = data_loss + alpha * (pde_loss + bound_loss + init_loss)
    return total_loss


def bound_func(pred, data):
    y_bound = 0
    return pred - y_bound


def init_func(pred, data):
    u_initial = tf.convert_to_tensor(np.sin(np.pi * data[:, 0]), dtype=tf.float32)
    u_initial = tf.reshape(u_initial, (-1, 1))
    return (pred - u_initial)


# Define optimizer and epochs
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# epochs = 1000  # Or fewer with early stopping
# alpha, beta, gamma = .001, .001, .001

def train_step(X_train, y_train, X_PDE_train, X_bound_1_train, X_bound_2_train, X_init_train):
    with tf.GradientTape() as tape:
        loss = pinn_loss(
            model=new_model,
            X_train=X_train,
            y_train=y_train,
            X_PDE_train=X_PDE_train,
            X_bound_1_train=X_bound_1_train,
            X_bound_2_train=X_bound_2_train,
            X_init_train=X_init_train,
            bound_func=bound_func,
            init_func=init_func,
            alpha=alpha
        )
    # Compute gradients of loss w.r.t model weights
    gradients = tape.gradient(loss, new_model.trainable_variables)
    # Apply gradients to update model weights
    optimizer.apply_gradients(zip(gradients, new_model.trainable_variables))
    return loss

# Define ranges for alpha
alpha_values = [0.0]
best_alpha = None
best_regular_loss = float('inf')  # Track only the data loss for comparison
epochs = 1000

# Loop through all combinations of alpha
for alpha in alpha_values:
    # Reset model and optimizer for each combination
    new_model = tuner.hypermodel.build(best_hps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Early stopping parameters
    best_loss = float('inf')  # Full validation loss for early stopping
    wait = 0
    patience = 5

    print(f"Testing alpha={alpha}")

    # Training loop
    for epoch in range(epochs):
        # Training step
        train_loss = train_step(X_train, y_train, X_PDE_train, X_bound_1_train, X_bound_2_train, X_init_train)

        # Full validation loss (includes PDE and regularization terms)
        val_loss = pinn_loss(
            model=new_model,
            X_train=X_val,
            y_train=y_val,
            X_PDE_train=X_PDE_val,
            X_bound_1_train=X_bound_1_val,
            X_bound_2_train=X_bound_2_val,
            X_init_train=X_init_val,
            bound_func=bound_func,
            init_func=init_func,
            alpha=alpha
        ).numpy()

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Calculate regular loss (e.g., MSE) for this hyperparameter combination
    regular_loss = tf.reduce_mean(tf.square(tf.reshape(new_model(X_val), [-1]) - tf.cast(y_val, tf.float32)))

    # Compare models based on regular loss
    if regular_loss < best_regular_loss:
        best_regular_loss = regular_loss
        best_alpha = alpha
        print(f"New best hyperparameters: alpha={best_alpha} "
              f"regular_loss={best_regular_loss}")

# Final results
print(f"Best hyperparameters: alpha={best_alpha}with regular loss={best_regular_loss}")

# Combine training and validation data
X_combined = tf.concat([X_train, X_val], axis=0)
y_combined = tf.concat([y_train, y_val], axis=0)

# Build and train the model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 1000
patience = 20
best_loss = float('inf')
wait = 0

for epoch in range(epochs):
    # Training step
    train_loss = train_step(X_combined, y_combined, X_PDE_train, X_bound_1_train, X_bound_2_train, X_init_train)

    # Compute full training loss for monitoring
    full_loss = pinn_loss(
        model=final_model,
        X_train=X_combined,
        y_train=y_combined,
        X_PDE_train=X_PDE_train,
        X_bound_1_train=X_bound_1_train,
        X_bound_2_train=X_bound_2_train,
        X_init_train=X_init_train,
        bound_func=bound_func,
        init_func=init_func,
        alpha=best_alpha
    ).numpy()

    # Early stopping
    if full_loss < best_loss:
        best_loss = full_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# Evaluate on test data
test_regular_loss = tf.reduce_mean(tf.square(tf.reshape(final_model(X_test), [-1]) - tf.cast(y_test, tf.float32))).numpy()
print(f"Test regular loss (MSE): {test_regular_loss}")