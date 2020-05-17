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


