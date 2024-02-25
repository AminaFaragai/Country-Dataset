#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#for supressing warnings
import warnings
warnings.filterwarnings('ignore')
#feature scalling
from sklearn.preprocessing import StandardScaler
#for kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris


# In[4]:


#importing dataset and checking the dataset
data = pd.read_csv('country_data.csv')
data.head()


# # EXPLARATORY DATA ANALYSIS

# In[5]:


#checkking out information about the datasets
data.info()


# In[6]:


#converting them to their actual values
data[['exports', 'health', 'imports']] = data[['exports', 'health', 'imports']].apply(lambda x : x*data["gdpp"]/100)
data.head()                                                                          


# In[7]:


#statistical summary
data.describe()


# # DATA VISUALIZATION

# In[8]:


#visualising graphical representation using bivariate analysis
sns.pairplot(data)
plt.show()


# In[9]:


#corollation of the dataframe using heatmap
plt.figure(figsize = (12,8))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[10]:


# univariate analysis

fig=plt.subplots(figsize=(12, 10))

for i, feature in enumerate(data.drop('country', axis=1).columns):
    plt.subplot(6, 3, i+1)
    plt.subplots_adjust(hspace = 2.0)
    sns.distplot(data[feature])
    plt.tight_layout()


# In[11]:


#handling outliers (boxplot)
fig=plt.subplots(figsize=(12, 12))

for i, feature in enumerate(data.drop('country', axis=1).columns):
    plt.subplot(6, 3, i+1)
    plt.subplots_adjust(hspace = 2.0)
    sns.boxplot(data[feature])
    plt.tight_layout()


# In[12]:


#updated data
data


# In[13]:


#scaling the data
standard_scaler = StandardScaler()
country_df_scaled = standard_scaler.fit_transform(data.iloc[:, 1:])


# # BUILDING THE MODEL

# In[18]:


#finding the optimal number of clusters with the elbow method
ssd = []
num_of_clusters = list(range(2,10))

for n in num_of_clusters:
    km = KMeans(n_clusters = n, max_iter = 50, random_state=101).fit(country_df_scaled)
    ssd.append(km.inertia_)
    
plt.plot(num_of_clusters, ssd, marker='o')

for xy in zip(num_of_clusters, ssd):    
    plt.annotate("point",(1,3),size=20, color = "red")
    
plt.xlabel("Number of clusters")
plt.ylabel("Inertia") # Inertia is within cluster sum of squares
plt.title("The Elbow method using inertia")
plt.show()


# In[19]:


#finding the number of clusters
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, max_iter=100)
    kmeans.fit(country_df_scaled)
    score = silhouette_score(country_df_scaled, kmeans.labels_)
    print("For cluster: {}, the silhouette score is: {}".format(i,score))


# In[17]:


#plotting graph for the silihouette value
silhouette_coefficients = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, max_iter=100,random_state=101 )
    kmeans.fit(country_df_scaled)
    score = silhouette_score(country_df_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.plot(range(2,10), silhouette_coefficients)
plt.xticks(range(2,10))
plt.xlabel("number of clusters")
plt.ylabel("Silhouette coefficient")
plt.show()


# # MODEL BUILDING

# In[36]:


#building kmeans model with 3 clusters
km = KMeans(n_clusters = 3, max_iter = 100, random_state=101)
km.fit(country_df_scaled)


# In[37]:


print(km.labels_)
print(km.labels_.shape)


# In[22]:


#updating data
data_clustered = data.iloc[:,:]
data_clustered = pd.concat([data_clustered,pd.DataFrame(km.labels_, columns=['cluster_id_km'])], axis = 1)
data_clustered.head()


# In[23]:


#checking numbers of km 
print(data_clustered['cluster_id_km'].value_counts())


# In[24]:


#listing down the countries on how it was clustered by KMeans model
print("Cluster 0 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 0].country.unique())

print("Cluster 1 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 1].country.unique())

print("Cluster 2 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 2].country.unique())


# In[25]:


# Final list of under-developed countries, in order of socio-economic condition from worst to better -

data_clustered[(data_clustered['cluster_id_km']==1)].sort_values(by=['gdpp', 'income', 'child_mort'], ascending=[True, True, False])[['country']].head(15)


# In[21]:


#building kmeans model with 4 clusters
km = KMeans(n_clusters = 4, max_iter = 100, random_state=101)
km.fit(country_df_scaled)


# In[44]:


print(km.labels_)
print(km.labels_.shape)


# In[45]:


#updating data
data_clustered = data.iloc[:,:]
data_clustered = pd.concat([data_clustered,pd.DataFrame(km.labels_, columns=['cluster_id_km'])], axis = 1)
data_clustered.head()


# In[46]:


#checking numbers of km 
print(data_clustered['cluster_id_km'].value_counts())


# In[28]:


#listing down the countries on how it was clustered by KMeans model
print("Cluster 0 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 0].country.unique())

print("Cluster 1 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 1].country.unique())

print("Cluster 2 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 2].country.unique())

print("Cluster 3 of KMeans model")
print(data_clustered[data_clustered['cluster_id_km'] == 3].country.unique())


# In[29]:


# Final list of under-developed countries, in order of socio-economic condition from worst to better -

data_clustered[(data_clustered['cluster_id_km']==1)].sort_values(by=['gdpp', 'income', 'child_mort'], ascending=[True, True, False])[['country']].head(15)


# In[ ]:




