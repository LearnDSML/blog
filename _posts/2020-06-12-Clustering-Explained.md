---
title: "Clustering Explained"
categories: tutorial
tags: [clustering, ml, algorithms]
---


## Clustering

On supervised learning (e.g. linear regression, decision tree), we already know what we are going to predict (target class, y), such as whether tomorrow will be raining or not. On the contrary, just like its name, unsupervised learning algorithms attempt to find pattern from the data - without knowing the labels (target class, y). One of unsupervised algorithms is clustering. 


So, Clustering is the task of grouping data into two or more groups based on the properties of the data, and more exactly based on certain patterns which are more or less obvious in the data. The goal is to find those patterns in the data that help us be sure that, given a certain item in our dataset, we will be able to correctly place the item in a correct group, so that it is similar to other items in that group, but different from items in other groups.
That means the clustering actually consists of two parts: one is to identify the groups and the other one is to try as much as possible to place every item in the correct group.

The ideal result for a clustering algorithm is that two items in the same group are as similar to each other, while two items from different groups are as different as possible.

![clustering](https://www.tutorialspoint.com/machine_learning_with_python/images/clustering_system.jpg)

It can be used for various things:
- ✔ Customer segmentation for the purpose of marketing.
- ✔ Customer purchasing behavior analysis for promotions and discounts.
- ✔ Identifying geo-clusters in an epidemic outbreak such as COVID-19.

A real-world brief example would be customer segmentation. As a business selling various type of products, it would be very difficult to find the perfect business strategy for each and every customer. But we can be smart about it and try to group our customers into a few subgroups, understand what those customers all have in common and adapt our business strategy for every group. Coming up with the wrong business strategy to a customer would mean perhaps losing that customer, so it's important that we've achieved a good clustering of our market.



### What is the difference between Clustering and Classification
**Classification** is the result of supervised learning which means that there is a known label that you want the system to generate.
For example, if you built a fruit classifier, it would say “this is an orange, this is an apple”, based on you showing it examples of apples and oranges.


**Clustering** is the result of unsupervised learning which means that you’ve seen lots of examples, but don’t have labels.
In this case, the clustering might return with “fruits with soft skin and lots of dimples”, “fruits with shiny hard skin” and “elongated yellow fruits” based not merely showing lots of fruit to the system, but not identifying the names of different types of fruit. Moreover, they are called clusters

![clusteringvsclassifiation](https://github.com/LearnDSML/blog/blob/master/assets/img/Clustering%26clasification-Animales.webp?raw=true)

![clustering](https://github.com/LearnDSML/blog/blob/master/assets/img/ClassificationvsClustering.png?raw=true)

### Types of Clustering

Given the subjective nature of clustering tasks, there are various algorithms that suit different types of problems. Each algorithm has its own rules and the mathematics behind how clusters are calculated. Clustering can be of many types based on Cluster Formation Methods. Followings are some other cluster formation methods −

#### Density-based
In these methods, the clusters are formed as the dense region. The advantage of these methods is that they have good accuracy as well as good ability to merge two clusters. Ex. Density-Based Spatial Clustering of Applications with Noise (DBSCAN), Ordering Points to identify Clustering structure (OPTICS) etc.

#### Hierarchical-based
In these methods, the clusters are formed as a tree type structure based on the hierarchy. They have two categories namely, Agglomerative (Bottom up approach) and Divisive (Top down approach). Ex. Clustering using Representatives (CURE), Balanced iterative Reducing Clustering using Hierarchies (BIRCH) etc.

#### Partitioning
In these methods, the clusters are formed by portioning the objects into k clusters. Number of clusters will be equal to the number of partitions. Ex. K-means, K-mediodes, PAM, Clustering Large Applications based upon randomized Search (CLARANS) etc.

#### Grid
In these methods, the clusters are formed as a grid like structure. The advantage of these methods is that all the clustering operation done on these grids are fast and independent of the number of data objects. Ex. Statistical Information Grid (STING), Clustering in Quest (CLIQUE).


A broad Classification of types of Clustering Algorithms is given below.

![types_of_clustering](https://github.com/LearnDSML/blog/blob/master/assets/img/types_of_clustering.png?raw=true)



### Clustering Performance Evaluation

There are various functions with the help of which we can evaluate the performance of clustering algorithms. Following are some important and mostly used functions given by the Scikit-learn for evaluating clustering performance −

#### Adjusted Rand Index
Rand Index is a function that computes a similarity measure between two clustering. For this computation rand index considers all pairs of samples and counting pairs that are assigned in the similar or different clusters in the predicted and true clustering. Afterwards, the raw Rand Index score is ‘adjusted for chance’ into the Adjusted Rand Index score by using the following formula −

> AdjustedRI=(RI−Expected−RI)/(max(RI)−Expected−RI)

Perfect labeling would be scored 1 and bad labelling or independent labelling is scored 0 or negative.

#### Mutual Information Based Score
Mutual Information is a function that computes the agreement of the two assignments. It ignores the permutations. There are following versions available −

#### Normalized Mutual Information (NMI)

#### Adjusted Mutual Information (AMI)
#### Fowlkes-Mallows Score
The Fowlkes-Mallows function measures the similarity of two clustering of a set of points. It may be defined as the geometric mean of the pairwise precision and recall.

Mathematically,

> FMS=TP/√(TP+FP)(TP+FN)

Here, TP = True Positive − number of pair of points belonging to the same clusters in true as well as predicted labels both.

FP = False Positive − number of pair of points belonging to the same clusters in true labels but not in the predicted labels.

FN = False Negative − number of pair of points belonging to the same clusters in the predicted labels but not in the true labels.

#### Silhouette Coefficient
The Silhouette function will compute the mean Silhouette Coefficient of all samples using the mean intra-cluster distance and the mean nearest-cluster distance for each sample.

Mathematically,

> S=(b−a)/max(a,b)
Here, a is intra-cluster distance.

and, b is mean nearest-cluster distance.


#### Contingency Matrix
This matrix will report the intersection cardinality for every trusted pair of (true, predicted). Confusion matrix for classification problems is a square contingency matrix.
