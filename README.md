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

Since the clustering methods use distance as a metric, data needs to be standardized.

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
**Cluster labels by KMeans -**

        **1    66**
        
        **0    61**
        
        **2    51**

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

**Cluster labels by Agglomerative -**

              **0    68**
              
              **2    59**
              
              **1    51**

### 3D Plot visualization of Agglomerative Clustering -
The clusters identified by Agglomerative can be visualized using a 3D plot.

![3D Agglomerative](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_agglo3d.png)

### Dendrograms -
A dendrogram is a diagram representing a tree. This diagrammatic representation is frequently used in different contexts. In hierarchical clustering, you categorize the objects into a hierarchy similar to a tree-like diagram which is called a dendrogram.
once one large cluster is formed by the combination of small clusters, dendrograms of the cluster are used to actually split the cluster into multiple clusters of related data points.

![Dendrogram](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_dendrogram.png)

**KMeans clustering is the best method in this case as the inertia of KMeans clustering is lesser compared to Agglomerative clustering**

### Principal Component Analysis - 
In simple words, principal component analysis is a method of extracting important variables (in form of components) from a large set of variables available in a data set. It extracts low dimensional set of features from a high dimensional data set with a motive to capture as much information as possible. With fewer variables, visualization also becomes much more meaningful.

When there are lot of variables aka features n(> 10) , then we are advised to do PCA. PCA is a statistical technique which reduces the dimensions of the data and help us understand, plot the data with lesser dimension compared to original data. As the name says PCA helps us compute the Principal components in data. Principal components are basically vectors that are linearly uncorrelated and have a variance with in data. From the principal components top p is picked which have the most variance.

![PCA](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_pca.png)

# Modelling:

## CROSS VALIDATION:
Cross validation is a powerful tool that is used for estimating the predictive power of your model, and it performs better than the conventional training and test set. Using cross validation, we can create multiple training and test sets and average the scores to give us a less biased metric.

**K-Fold Cross Validation:** Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation. Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

From the KFold Cross Validation, The model that has low bias error and variance error is Random Forest. Since it is an overfitting model, we can take the next model which has low variance and bias error. Gradient Boosting Classifier is the best model both in terms of accuracy and bias and variance error.


## Logistic Regression -
Logistic regression is a statistical method for analysing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.

## Decision Tree Classifier -
Linear regression and logistic regression models fail in situations where the relationship between features and outcome is nonlinear or where features interact with each other. Time to shine for the decision tree! Tree based models split the data multiple times according to certain cut-off values in the features. Through splitting, different subsets of the dataset are created, with each instance belonging to one subset. The final subsets are called terminal or leaf nodes and the intermediate subsets are called internal nodes or split nodes. To predict the outcome in each leaf node, the average outcome of the training data in this node is used. Trees can be used for classification and regression. There are various algorithms that can grow a tree. They differ in the possible structure of the tree (e.g. number of splits per node), the criteria how to find the splits, when to stop splitting and how to estimate the simple models within the leaf nodes. The classification and regression trees (CART) algorithm is probably the most popular algorithm for tree induction. We will focus on CART, but the interpretation is similar for most other tree types.

### K Nearest Neighbour Classifier -

K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification predictive problems in industry. The following two properties would define KNN well −

1. Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.

2. Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.

### Bagging -
Bagging stands for bootstrap aggregation. One way to reduce the variance of an estimate is to average together multiple estimates. Bagging uses bootstrap sampling to obtain the data subsets for training the base learners. For aggregating the outputs of base learners, bagging uses voting for classification and averaging for regression.

### AdaBoost - 
Boosting refers to a family of algorithms that are able to convert weak learners to strong learners. The main principle of boosting is to fit a sequence of weak learners− models that are only slightly better than random guessing, such as small decision trees− to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.
The predictions are then combined through a weighted majority vote (classification) or a weighted sum (regression) to produce the final prediction. The principal difference between boosting and the committee methods, such as bagging, is that base learners are trained in sequence on a weighted version of the data.

AdaBoost is one of the first boosting algorithms to be adapted in solving practices. Adaboost helps you combine multiple “weak classifiers” into a single “strong classifier”. Here are some (fun) facts about Adaboost!

1. The weak learners in AdaBoost are decision trees with a single split, called decision stumps.

2. AdaBoost works by putting more weight on difficult to classify instances and less on those already handled well.

3. AdaBoost algorithms can be used for both classification and regression problem.

### Naives Bayes -
It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. NaiveBayes is a probabilistic classifier which returns the probability of a test point belonging to a class rather than the label of the test point

### Random Forest Classifier -
This is a classifier that evolves from decision trees. It actually consists of many decision trees. To classify a new instance, each decision tree provides a classification for input data; random forest collects the classifications and chooses the most voted prediction as the result. The input of each tree is sampled data from the original dataset. In addition, a subset of features is randomly selected from the optional features to grow the tree at each node. Each tree is grown without pruning. Essentially, random forest enables a large number of weak or weakly-correlated classifiers to form a strong classifier.

## Gradient Boosting Classifier -
Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. 

![Bias Variance Error](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_bias_var.png)

**KNN performs well as there is a good bias error and variance error trade-off compared to other models.**

## Important Features from the Model -
Those variables which have positive or negative higher coeeficients are the most important features of the model. 
The important features of the model are

![Important Features](https://github.com/SaranyaDScientist/Wine/blob/master/Wine_feat.png)






