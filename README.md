Machine Learning Project: Country Clustering Analysis

Introduction

This machine learning project aims to implement the K-means clustering algorithm on the provided country dataset. The Elbow method and the silhouette method were utilized to determine the optimum number of clusters. Both 3 and 4 clusters were explored to compare the model, resulting in similar outcomes.

Problem Statement
The dataset 'countries-data' contains information about developmental indices of numerous countries. Categorizing countries based on socio-economic and health factors can help determine the overall development of each country.

Importing Modules and Reading the Data
The data was imported from 'country-data.csv' for analysis. Information about child mortality, exports, health, imports, income, inflation, life expectancy, total fertility, and GDP per capita of several countries were contained in the file. Clustering the countries will group them based on similar levels of development, allowing identification of the least developed and most developed countries.

Data Visualization
Pairplot was used to visualize the features of the data, showcasing linear relationships and distributions within the dataset. Correlation analysis using a heatmap revealed associations between different features.

Model Building using K-means Clustering Algorithm
The elbow method of K-means clustering was used to cluster all the data, with high correlations observed between import and export of countries, child mortality and life expectancy, health and GDP per capita, income and GDP per capita, and child mortality and total fertility. Distplot and boxplot were used to understand the skewness of variables and identify outliers, respectively.

Handling Outliers
Outliers were identified using boxplots, and the optimal number of clusters for the given data was determined to be 4.

Comparing the Models
Models built using K-means with 3 and 4 clusters produced similar outputs, indicating the model's effectiveness. The top driving factors for clustering were identified as GDP per capita, income, and child mortality.

Conclusion
Low GDP per capita and income imply a high rate of child mortality.
Life expectancy in under-developed countries remained consistent across both 3 and 4 clusters.
The top 15 under-developed countries were found to be the same in both clusters.
