sample_sizes = [150, 175, 250, 350, 500]
n_bad_values = []
len_lower_values = []

p_values = {}

n_bad = len(max_diffs[max_diffs < 0])
prob = 1 - binom.cdf(n_bad - 1, len(max_diffs), alpha)
p_values[num_points] = prob
print(f"n_new: {len(lower)}, P(X >= {n_bad}) = {prob}")
n_bad_values.append(n_bad)
len_lower_values.append(len(lower))  # This is the test sample size


# Convert p_values dict to lists for plotting
p_values_x = list(p_values.keys())
p_values_y = list(p_values.values())

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# Plot 1: n_bad vs len(lower)
plt.subplot(1, 2, 1)
sns.lineplot(x=len_lower_values, y=n_bad_values, marker='o', linestyle='-', color='b')
plt.xlabel("Conformal Test Sample Size (15% of original sample size)")
plt.ylabel("n_bad (Number of nonconformal points)")
plt.title("n_bad (Number of nonconformal points) vs n_new")
plt.grid(True)

# Plot 2: p-value vs original sample size
plt.subplot(1, 2, 2)
sns.lineplot(x=p_values_x, y=p_values_y, marker='o', linestyle='-', color='r')
plt.xlabel("Original Sample Size")
plt.ylabel("p-beta")
plt.title("p-beta vs Original Sample Size")
plt.ylim(bottom=-0.05)
plt.grid(True)

plt.tight_layout()
plt.show()