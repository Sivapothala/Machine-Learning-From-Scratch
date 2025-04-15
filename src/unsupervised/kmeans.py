import numpy as np
from ..fundamentals.linear_algebra import LinearAlgebra

class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        
    def fit(self, X):
        """Train the K-means model"""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iterations):
            # Assign samples to nearest centroids
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
    def _assign_clusters(self, X):
        """Assign each sample to the nearest centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X) 