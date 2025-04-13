"""
EM is efficient for Gauss distributions, but not for spiral non convex data.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# Load data
file_path = os.path.join(os.path.dirname(__file__), 'spiral.txt')
data = np.loadtxt(file_path)
X = data[:, :2]
true_labels = data[:, -1].astype(int)

# Parameter search range
components_range = range(2, 11)  # Try 2 to 10 clusters

best_ari = -1
best_k = None

# Try different number of components
for k in components_range:
    start = time.time()
    gmm = GaussianMixture(n_components=k, random_state=0)
    labels = gmm.fit_predict(X)
    end = time.time()

    ari = adjusted_rand_score(true_labels, labels)
    print(f"n_components = {k} → ARI = {ari:.4f} | time = {end - start:.2f}s")

    if ari > best_ari:
        best_ari = ari
        best_k = k

# Final result
print(f"\nBest ARI = {best_ari:.4f} at n_components = {best_k}")

# Optional: visualize best clustering
gmm = GaussianMixture(n_components=best_k, random_state=0)
labels = gmm.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='tab10')
plt.title(f"GMM clustering (k={best_k})")
plt.show()

