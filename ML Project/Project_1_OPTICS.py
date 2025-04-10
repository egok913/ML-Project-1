import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

file_path = "C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\pathbased.txt"

# We separately store the first two columns as features and the third column as labels. Data types are also changed from string to float and integer respectively. 
features = []
labels = []
with open(file_path, 'r') as f:
    for l in f:
        parts = l.strip().split()
        c1 = float(parts[0])
        c2 = float(parts[1])
        label = int(parts[2])
        features.append([c1, c2])
        labels.append(label)


### OPTICS Clustering ###
# We create an instance of the OPTICS class, fit our data into this model, and finally predict the labels of each data point based on this model.
optics = OPTICS(min_samples=4, xi=0.05)
labels_opt = optics.fit_predict(features)

# Separating the two features from tuples
x = [pt[0] for pt in features]
y = [pt[1] for pt in features]

# Plotting the true labels provided with the dataset
plt.scatter(x, y, c=labels, marker='o')
plt.title('True Labels')
plt.show()

# Plotting OPTICS clustering results
plt.scatter(x, y, c=labels_opt, marker='o')
plt.title('OPTICS Clustering')
plt.show() 

# Evaluating clustering results
score = silhouette_score(features, labels_opt)
ari = adjusted_rand_score(labels, labels_opt)
print(f"OPTICS Clustering Silhouette Score: {score}")
print(f"Adjusted Random Index: {ari}")