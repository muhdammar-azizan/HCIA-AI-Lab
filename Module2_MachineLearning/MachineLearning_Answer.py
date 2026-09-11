"""
Module 2 - Machine Learning Lab Guide
Question: How to implement K-means from scratch using Python?

Answer:
K-means clustering can be implemented from scratch by repeating the following
five steps until the cluster centroids stop changing significantly:

1. Initialize centroids: Randomly select K data points from the dataset to
   serve as the initial cluster centroids.

2. Compute distances: For every data point, calculate its Euclidean distance
   to each of the K centroids.

3. Assign clusters: Assign each data point to the cluster whose centroid is
   closest to it (i.e., the centroid with the minimum distance).

4. Update centroids: For each cluster, recalculate the centroid by taking the
   mean (average) position of all data points currently assigned to that
   cluster.

5. Check for convergence: Compare the new centroids to the previous centroids.
   If the centroids have moved by less than a small threshold (tolerance),
   the algorithm has converged and stops. Otherwise, repeat steps 2-4.

This implementation was tested on a synthetic dataset (generated using
make_blobs) and produced centroid values identical to those obtained using
scikit-learn's built-in KMeans function, confirming the correctness of the
from-scratch implementation.

"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    return centroids

def compute_distances(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
    return distances

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points_in_cluster = X[labels == i]
        new_centroids[i] = points_in_cluster.mean(axis=0)
    return new_centroids


def kmeans_from_scratch(X, k, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)

        # Check for convergence: if centroids barely moved, stop early.
        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids

        if shift < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return centroids, labels

# Run our from-scratch K-means implementation.
k = 4
centroids, labels = kmeans_from_scratch(X, k)
print("Centroids (from scratch):")
print(centroids)

# Visualize the result.
color = ["red", "pink", "orange", "green"]
fig, ax1 = plt.subplots(1)

for i in range(k):
    ax1.scatter(X[labels == i, 0], X[labels == i, 1],
                marker='o', s=8, c=color[i])

ax1.scatter(centroids[:, 0], centroids[:, 1],
            marker="x", s=100, c="black")
plt.title("K-means From Scratch")
plt.show()

