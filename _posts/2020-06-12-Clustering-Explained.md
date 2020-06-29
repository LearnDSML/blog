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

![clustering](https://programmerbackpack.com/content/images/2020/04/Cluster-example.png)

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

![clustering](https://techdifferences.com/wp-content/uploads/2018/01/Untitled.jpg)

### Types of Clustering

Given the subjective nature of clustering tasks, there are various algorithms that suit different types of problems. Each algorithm has its own rules and the mathematics behind how clusters are calculated.

![clustering](https://github.com/LearnDSML/blog/blob/master/assets/img/types_of_clustering.png)
