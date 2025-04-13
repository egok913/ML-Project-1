"""
DBSCAN works well for non-convex shapes because it connects points by density, not by geometry.  
"""


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# Load the spiral dataset
file_path = os.path.join(os.path.dirname(__file__), 'spiral.txt')
data = np.loadtxt(file_path)

X = data[:, :2]
true_labels = data[:, -1].astype(int)

# Parameter ranges to try
eps_range = np.arange(0.5, 2.1, 0.1)        # eps values from 0.5 to 2.0
min_samples_range = range(2, 6)             # min_samples values from 2 to 5

best_ari = -1
best_params = (None, None)

# Try all combinations of eps and min_samples
for eps in eps_range:
    for min_samples in min_samples_range:
        start = time.time()
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        end = time.time()

        ari = adjusted_rand_score(true_labels, labels)
        print(f"eps={eps:.2f}, min_samples={min_samples} → ARI={ari:.4f} | time={end - start:.2f}s")

        if ari > best_ari:
            best_ari = ari
            best_params = (eps, min_samples)

# Summary of best result
print(f"\nBest ARI = {best_ari:.4f} at eps = {best_params[0]:.2f}, min_samples = {best_params[1]}")

# Visualize best clustering
db = DBSCAN(eps=best_params[0], min_samples=best_params[1])
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='tab10')
plt.title(f"Best DBSCAN clustering (eps={best_params[0]:.2f}, min_samples={best_params[1]})")
plt.show()
