# Clustering is similar to classification, but the basis is different.
# In Clustering we don't know what we're looking for, and we are trying to identify some segments or clusters in our data.
# When we use clustering algorithms on our dataset, unexpected things can suddenly pop up like structures, clusters and groupings we would have never thought of otherwise.
# we will discuss:

## 1. K-Means Clustering
## 2. Hierarchical Clustering

# Clustering can be defined as grouping unlabeled data
# So far in this course we've been working with supervised learning types of algorithms, which includes regression and classification
# The way supervised learning works, is that we already have some training data and answers in that training data that we supply to the model. (eg input as images of apples plus labels and output: new entry which is going to tell you if it's an apple)
# the unsupervised learning: we don't have answers and the model has to think for itself (eg. input data for images-fruits without any labels -> the model analyses the data and will group fruits into categories)
# unsupervised learning will only group things (eg pictures of fruits) into it's own categories. (so it won't know if it's a banana, apple). Will just see certain similarities/differences in the data