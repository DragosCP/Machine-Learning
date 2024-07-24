# HC clustering. types:
#   Divisive
#   Agglomerative: bottom-up approach - we will focus on this one

# Agllomerative: 
# Step 1: Make each data point a single-point cluster -> That forms N clusters
# Step 2: Take the 2 closest data points and make them one cluster -> That forms N-1 clusters
# Step 3: Take the 2 closest clusters and make them one cluster -> That forms N-2 clusters
# Step 4: Keep repeating until there is only one cluster (repeat step 3)
# finish.

# Basically the way the gierarchical custering algorithm works is that it maintains a memory of how we went through these steps and that memory is stored in a Dendograms

# Closest clusters : Euclidean Distance is one of the formula
# distance between P1 and P2 = square root of [ (X2-X1)squared  + (y2-y1)squared ]

# Distance between two clusters: 
#   Option 1: closest points
#   Option 2: furthest points
#   Option 3: Average distance
#   Option 4: Distance between Centroids

# Dendograms (graphs in course) - memory of the hierarchical clustering algorithm
# we can set up a similarity tresh hold, at a specific point in the Dendrogram, let's say 2, anything that goes above (that will have dissimilarity) will be ignored, so we will end up in a optimal number of clusters.
# Once we create the dendrogram and we visualise it, we compare the Euclidean distances (on the y axis, vertically) between a horisontal bar up to the next horisontal line. And then we compare them. The largest vertically line (euclidean distance) out of these will be our tresh holder (paralel with X and will cross our Dendrogram that will tell us how many clusters will have)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


database = pd.read_csv(r"03_Clustering\Mall_Customers.csv")
X = database.iloc[:, [3, 4]].values

# using the Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
# hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
hc = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
# print(y_hc)

# Visualising the clusters. 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='cluster3')
# plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='cluster4')
# plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()