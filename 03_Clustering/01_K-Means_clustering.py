# K-Means clustering

# we decide the number of clustering, then, for each cluster we randomly place a centriod and the K-Means will assign each of the data points to the closest centroid. We will be adding an equidistant line and anything above will go to one centroid and the same for the below line.

# we also need to calculate the centre of mass (gravity) for each point (x and y) of each clusters (without centroild, of course). We will plot all the points to X and Y coordinates, and take the average for each cluster and then re-assign the centroid to that average. And then we draw the equidistant line again. and repeat UNTIL we're drawing the perfect equidistant line that splits the results properly.

# The Elbow Method: will help us decide how many clusters to identify in our data. This method applies after we run the K-Means algorithm  

# K-Means++ is design to combat Random Initialization Trap, where we apply K-Means again to a pre-selected centroids (with K-Means already applied) which will end up in different clusters and different results.
# K-Means++ initialization algoritthm:
    ## Step 1: Choose first centroid at random among data points
    ## Step 2: For each of the remaining data points compute the distance (D) to the nearest out of already selected centroids
    ## Step 3: Choose next centroid amonth remaining data points using weighted random selection: weighted by D squared.
    ## Step 4: Repeat Steps 2 and 3 until all k centroids have been selected
    ## Step 5: Proceed with standard k-means clustering

# problem: identify patterns in mall's customers.
# we will create a dependent variable, which will take a finite number of values, let's say 4 or 5, and each of the values will be a class of this dependent variable we're going to create.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


database = pd.read_csv(r"03_Clustering\Mall_Customers.csv")
X = database.iloc[:, [3, 4]].values
# we only bringing the values of Annual income feature and Spending score feature to plot the results in 2D
# also, we don't have to split the dataset into training and test set as this implias of having a dependent variable vector which we don't have at the moment, we will create one once we train the model on the dataset.


# Using the Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # training the k-means alg
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() # we can see that from number 5 to 10, the line is getting almost flat, so 5 will be our number of clusters.


# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

# Visualising the clusters. 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = "Centroids")
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
