# Aggregation (Kmeans:30, EM:50, DBSCAN: eps=1.07, min_samples=5, OPTICS: min_samples=6, xi=0.12): "C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\Aggregation.txt"
# Pathbased1 (DBSCAN(eps=1.6, min_samples=7)) : "C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\pathbased.txt"


import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

file_path = input("Path to dataset file: ")
cluster_n = int(input("Number of clusters: "))

data = np.loadtxt(file_path, delimiter=' ')
data = data.reshape(-1, 2)  # Convert data into a 2D array. We can use -1 because we don't need to specify an exact number for both dimensions.


### EM Clustering ###
# We create an instance of the GaussianMixture class, fit out data into this model, and finally predict the labels of each data point based on this model.
em = GaussianMixture(n_components=cluster_n)
em.fit(data)
labels = em.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o')
plt.title('EM Clustering (GMM)')
plt.show()   

# Evaluating clustering results
score = silhouette_score(data, labels)
print("EM Clustering Silhouette Score:", score)


### KMeans Clustering ###
# We create an instance of the KMeans class, fit out data into this model, and finally predict the labels of each data point based on this model.
kmeans = KMeans(n_clusters=cluster_n)
kmeans.fit(data)
labels_km = kmeans.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=labels_km, marker='o')
plt.title('KMeans Clustering')
plt.show() 

# Evaluating clustering results
score = silhouette_score(data, labels_km)
print("KMeans Clustering Silhouette Score:", score)


