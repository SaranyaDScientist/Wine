# Wine Classification (Through Clustering)

The wine dataset contains the results of a chemical analysis of wines grown in a specific area of Italy. Three types of wine are represented in the 178 samples, with the results of 13 chemical analyses recorded for each sample. 

### Source - https://archive.ics.uci.edu/ml/datasets/wine

# Data Understanding/Exploratory Data Analysis:
Once the data have been collected, we have to understand each and every variable of the data and its characteristics. This can be done by checking number and type of features, descriptive statistics and visualizations, missing values, inconsistent data records etc.

**Shape of the dataset – 178 Rows & 13 Columns**

**Type of features – 1 Object & 12 Integers**

**Target Variable - Cultivator (object) - [1,2,3] - Multiple Classification**

### Most of the real world data are unlabelled and unclassified. There comes the implementation of Unsupervised Machine Learning. So, lets remove the target column 'Cultivator' and perform clustering

**Missing Values – No missing values**

**Descriptive Statistics -**
![Descriptive Statistics](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_desc_stats.png)

### Data Visualization (Univariate Analysis) - 
From the Univariate analysis, we can understand the central tendency and spread of numerical variables and the proportion of the various levels of categorical variables. Here, numerical variables are analysed through  **Box plots**.

### Outliers -
Outliers are data points that are far from other data points. In other words, they are unusual values in a dataset. In this case, there are outliers and since the outliers are less than 5%, it is imputed with the median value.

**Outlier Detection -**
![Outliers](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_outlier.png)

### Data Visualization (Multivariate Analysis) - 
From the **pairplot**, the impact of various X variables on Y variable are visualized, thereby giving clues for feature selection.

**Pairplot -**
![Pairplot](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_pairplot.png)

A heatmap is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions. And using **heatmap**, the correlation between the variables are known. From that, we can also find out the highly correlated features.

**Heatmap -**
![Heatmap](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_corr.png)

### Standardization -
Data standardization is the process of rescaling one or more attributes so that they have a mean value of 0 and a standard deviation of 1. Standardization assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to be true, but the technique is more effective if your attribute distribution is Gaussian.

### Z - Score - 
Simply put, a z-score (also called a standard score) gives you an idea of how far from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is. A z-score can be placed on a normal distribution curve. Z-scores range from -3 standard deviations (which would fall to the far left of the normal distribution curve) up to +3 standard deviations (which would fall to the far right of the normal distribution curve). In order to use a z-score, you need to know the mean μ and also the population standard deviation σ.

### Unsupervised Learning -
 Unsupervised learning is where you only have `input data (X)` and `no corresponding output variables`. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because unlike supervised learning above there is `no correct answers` and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data.

### Clustering -
Clustering is the process of dividing the entire data into groups (also known as clusters) based on the patterns in the data.
In clustering, we do not have a target to predict. We look at the data and then try to club similar observations and form different groups. Hence it is an unsupervised learning problem. Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). Clustering can be considered the most important unsupervised learning problem; so, as every other problem of this kind, it deals with finding a structure in a collection of unlabeled data. A loose definition of clustering could be “the process of organizing objects into groups whose members are similar in some way”. A cluster is therefore a collection of objects which are “similar” between them and are “dissimilar” to the objects belonging to other clusters.

## KMeans Clustering -
The k-means clustering algorithm is known to be efficient in clustering large data sets. This algorithm is one of the simplest and best known unsupervised learning algorithm. The k-means algorithm aims to partition a set of objects, based on their attributes/features, into k clusters, where k is a predefined constant. The algorithm defines k centroids, one for each cluster. The centroid of a cluster is formed in such a way that it is closely related, in terms of similarity (where similarity can be measured by using different methods such as Euclidean distance or Extended Jacquard) to all objects in that cluster. Technically, what k-means is interested in, is the variance. It minimizes the overall variance, by assigning each object to the cluster such that the variance is minimized.

**Inertia of KMeans - 1243.80**
**Cluster labels by KMeans -
        1    66
        0    61                             
        2    51**

## Elbow Method -
The basic idea behind partitioning methods, such as k-means clustering, is to define clusters such that the total intra-cluster variation [or total within-cluster sum of square (WSS)] is minimized. The total WSS measures the compactness of the clustering and we want it to be as small as possible. The Elbow method looks at the total WSS as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn’t improve much better the total WSS.

![Elbow Method](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_elbow.png)

### 3D Plot visualization of KMeans Clustering -
The clusters identified by KMeans can be visualized using a 3D plot.

![3D KMeans](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_kmeans3d.png)

### Agglomerative Clustering -
It starts by calculating the distance between every pair of observation points and store it in a distance matrix. It then puts every point in its own cluster. Then it starts merging the closest pairs of points based on the distances from the distance matrix and as a result the amount of clusters goes down by 1. Then it recomputes the distance between the new cluster and the old ones and stores them in a new distance matrix. Lastly it repeats steps 2 and 3 until all the clusters are merged into one single cluster. Or In Simple words it can be stated as :-
 
Hierarchical clustering starts by treating each observation as a separate cluster.

Then, it repeatedly executes the following two steps:

(1) identify the two clusters that are closest together, and

(2) merge the two most similar clusters.

This continues until all the clusters are merged together.

Inter-cluster linkage methods for merging clusters:

There are several ways to measure the distance between clusters in order to decide the rules for clustering,
and they are often called Linkage Methods.

Some of the common linkage methods are:

**Complete-linkage:** calculates the maximum distance between clusters before merging.
**Single-linkage:** calculates the minimum distance between the clusters before merging. This linkage may be used to detect high values in your dataset which may be outliers as they will be merged at the end.
**Average-linkage:** calculates the average distance between clusters before merging.
**Centroid-linkage:** finds centroid of cluster 1 and centroid of cluster 2, and then calculates the distance between the two before merging.
The choice of linkage method entirely depends on you and there is no hard and fast method that will always give you good results. Different linkage methods lead to different clusters.

**inertia of the best model: average manhattan 1253.20**

**Cluster labels by Agglomerative -
              0    68
              2    59
              1    51**






