# Name: Abhay Singh Tomar
# BTech Cse, 3rd year
# sec:"B"
# Cu24250144
# ROll no. = 1

# Experiment No. 7

# Experiment Name
# Customer Segmentation using K-Means Clustering

# Aim
# To implement the K-Means Clustering algorithm on a real-world dataset for customer segmentation and analyze customer groups based on purchasing behavior and demographic characteristics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Upload CSV file
uploaded = files.upload()

# Read dataset
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("Dataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
display(df.describe())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

# Select important features
features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

X = df[features]

print("Selected Features:")
print(X.head())

print("\nShape after preprocessing:", X.shape)
df.head()

# Standardize the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert into DataFrame for easy viewing
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("Scaled Data:")
display(X_scaled_df.head())

wcss = []

# Calculate WCSS for different values of K
for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)

plt.show()

silhouette_scores = []

# Calculate silhouette score for K = 2 to 10
for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    
    print(
        "K =", k,
        " Silhouette Score =",
        round(score, 4)
    )

# Plot scores
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker='o'
)

plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)

plt.show()

# Select number of clusters
k = 5

# Create and train K-Means model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Predict cluster for each customer
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed!")
print("\nCustomer Cluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

print("\nSample Data:")
display(df.head(10))

# Cluster visualization
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100
)

# Plot cluster centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 1],
    centers[:, 2],
    marker='X',
    s=250,
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()

# Analyze clusters
print("\nAverage Characteristics of Each Cluster:")

cluster_summary = df.groupby('Cluster')[features].mean()

display(cluster_summary.round(2))


# Questions & Answers

# Q1. What is clustering? How does it differ from classification?

# Answer: Clustering is an unsupervised machine learning technique used to group similar data points into clusters. It does not require predefined labels. Classification is a supervised learning technique in which data is assigned to predefined classes using labelled training data.

# Q2. Explain the working principle of the K-Means Clustering algorithm.

# Answer: K-Means divides the dataset into a specified number of clusters (K). First, K centroids are selected randomly. Each data point is assigned to the nearest centroid, then the centroids are recalculated based on the mean of the assigned points. This process continues until the centroids become stable.

# Q3. Why is feature scaling important before applying K-Means clustering?

# Answer: Feature scaling makes all selected features contribute equally to clustering. K-Means uses distance calculations, so a feature with a larger numerical range can dominate the results. Standardization brings the features to a comparable scale and improves clustering performance.

# Q4. What is the Elbow Method, and how does it help determine the optimal number of clusters?

# Answer: The Elbow Method is used to find a suitable value of K in K-Means. It calculates the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters. The point where the decrease in WCSS starts becoming slower is considered the elbow and is generally selected as the optimal K.

# Q5. What is the Silhouette Score? How is it used to evaluate clustering performance?

# Answer: The Silhouette Score measures how well each data point fits within its assigned cluster compared with other clusters. Its value generally ranges from -1 to 1. A higher positive score indicates that the clusters are well separated and the data points are properly grouped.

# Q6. Why is K-Means considered an unsupervised machine learning algorithm?

# Answer: K-Means is considered unsupervised because it works without predefined class labels. It discovers natural groups or patterns in the dataset based on the similarity between data points. The clusters are created automatically according to the selected features.

# Q7. Mention any four real-world applications of customer segmentation.

# Answer: Four applications of customer segmentation are:

# Targeted marketing campaigns

# Personalized product recommendations

# Customer retention and loyalty programs

# Improving customer engagement

# Customer segmentation helps businesses understand different customer groups and create suitable strategies for them

# Q8. What are the limitations of the K-Means algorithm?

# Answer: K-Means requires the number of clusters to be selected in advance. It can be affected by the initial selection of centroids and outliers. It also works best when clusters are relatively compact and well separated. Different feature scales can also affect the results if scaling is not performed.

# Q9. How can businesses use customer segmentation to improve marketing and customer retention?

# Answer: Businesses can identify groups of customers with similar needs, purchasing behavior, or spending patterns. They can then create targeted offers, personalized advertisements, loyalty programs, and product recommendations for each group. This can improve customer engagement and help retain customers.

# Q10. Compare K-Means Clustering with Hierarchical Clustering based on their working principles and applications.

# K-Means Clustering	Hierarchical Clustering
# Divides data into a predefined number of clusters.	Builds a hierarchy of clusters.
# Requires the value of K.	Does not require K at the beginning.
# Uses centroids and distance calculations.	Uses distances to merge or split clusters.
# Generally faster for large datasets.	Can be computationally expensive for large datasets.
# Useful for customer segmentation and grouping large datasets.	Useful when hierarchical relationships between groups are important.
