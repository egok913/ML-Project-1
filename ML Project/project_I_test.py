# unbalance2 (random_state=10, 2): C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\unbalance2.txt
# skewed (random_state=3, 140): C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\skewed.txt
# overlap (random_state=1, 1): C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\overlap.txt
# asymmetric (random_state=1, 1): C:\SDU Courses\DSK804 Data Mining and Machine Learning\Project\asymmetric.txt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from a text file, assuming each data point has 2 integer coordinates separated by a space."""
    data = np.loadtxt(file_path, delimiter=' ')
    if data.ndim == 1:
        data = data.reshape(-1, 2)  # Ensure it's a 2D array
    return data

def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled)
    
    return labels

def plot_clusters(data, labels):
    """Plot the clustered data."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('DBSCAN Clustering')
    plt.colorbar()
    plt.show()

def main():
    file_path = input("Enter the path to your dataset file: ")
    eps = float(input("Enter the value of eps (e.g., 0.5): "))
    min_samples = int(input("Enter the value of min_samples (e.g., 5): "))
    
    data = load_data(file_path)
    labels = perform_dbscan_clustering(data, eps, min_samples)
    plot_clusters(data, labels)
    
    print("Cluster Labels:\n", labels)

if __name__ == "__main__":
    main()
