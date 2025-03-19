import numpy as np
import matplotlib.pyplot as plt

results = np.load("pvalue_results.npy", allow_pickle=True).item()

for beta, data in results.items():
    plt.hist(data['pvalues'], bins=20, alpha=0.7, label=f"Beta {beta}")

plt.xlabel("P-value")
plt.ylabel("Frequency")
plt.legend()
plt.title("P-value Distributions for Different Beta Values")
plt.show()