import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hierarchy

breastCancerData = pd.read_csv("breast_cancer.csv", header=None).values
mushroomsData = pd.read_csv("mushrooms.csv", header=0)
wineData = pd.read_csv("wineQualityReds.csv", header=0).values

encoder = LabelEncoder()
for column in mushroomsData.columns:
    mushroomsData[column] = encoder.fit_transform(mushroomsData[column])

breastCancerX = breastCancerData[:, :-1]
breastCancerLabels = breastCancerData[:, -1]
wineDataX = wineData[:, :-1]
wineLabels = wineData[:, -1]
mushroomX = mushroomsData.iloc[:, :-1].values
mushroomLabels = mushroomsData.iloc[:, -1].values

def bkmeans(X, k, iter):
    clusters = [X] #Start with 1 cluster

    while len(clusters) < k:
        largestCluster = max(clusters, key=len) #Get largest cluster
        kmeans = KMeans(n_clusters=2, random_state=0) #Use Kmean to divide into 2 sub clusters
        kmeans.fit(largestCluster)

        bestSSE = float('inf')
        bestLabels = None

        for _ in range(iter):
            kmeans.fit(largestCluster)
            sse = kmeans.inertia_
            if sse < bestSSE:
                bestSSE = sse
                bestLabels = kmeans.labels_

        #Split the largest cluster into two sub clusters
        newClusters = [largestCluster[bestLabels == 0], largestCluster[bestLabels == 1]]

        #Replace the largest cluster with the new sub clusters
        clusters.remove(largestCluster)
        clusters.extend(newClusters)

    #Assign cluster indices to each observation
    clusterIndicies = np.zeros(len(X), dtype=int)
    for i, cluster in enumerate(clusters):
        clusterIndicies[np.isin(X, cluster).all(axis=1)] = i

    return clusterIndicies

pca = PCA(n_components=2)
breastCancerXPCA = pca.fit_transform(breastCancerX)
wineDataXPCA = pca.fit_transform(wineDataX)
mushroomXPCA = pca.fit_transform(mushroomX)

#Bisecting k-Means
breastCancerbkmeans = bkmeans(breastCancerXPCA, k=2, iter=10)
wineDatabkmeans = bkmeans(wineDataXPCA, k=2, iter=10)
mushroombkmeans = bkmeans(mushroomXPCA, k=2, iter=10)

#Classic k-Means
breastCancerkmeans = KMeans(n_clusters=2, random_state=0).fit_predict(breastCancerXPCA)
wineDatakmeans = KMeans(n_clusters=2, random_state=0).fit_predict(wineDataXPCA)
mushroomkmeans = KMeans(n_clusters=2, random_state=0).fit_predict(mushroomXPCA)

#Hierarchical Clustering
breastCancerHierarchical = linkage(breastCancerXPCA, method='ward')
wineDataHierarchical = linkage(wineDataXPCA, method='ward')
mushroomHierarchical = linkage(mushroomXPCA, method='ward')


fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Plotting results for Breast Cancer dataset
axes[0, 0].scatter(breastCancerXPCA[:, 0], breastCancerXPCA[:, 1], c=breastCancerbkmeans)
axes[0, 0].set_title('Breast Cancer - Bisecting k-Means')
axes[0, 1].scatter(breastCancerXPCA[:, 0], breastCancerXPCA[:, 1], c=breastCancerkmeans)
axes[0, 1].set_title('Breast Cancer - Classic k-Means')
dendro = hierarchy.dendrogram(breastCancerHierarchical, ax=axes[0, 2])
axes[0, 2].set_title('Breast Cancer - Hierarchical Clustering')

# Plotting results for Mushrooms dataset
axes[1, 0].scatter(mushroomXPCA[:, 0], mushroomXPCA[:, 1], c=mushroombkmeans)
axes[1, 0].set_title('Mushrooms - Bisecting k-Means')
axes[1, 1].scatter(mushroomXPCA[:, 0], mushroomXPCA[:, 1], c=mushroomkmeans)
axes[1, 1].set_title('Mushrooms - Classic k-Means')
dendro = hierarchy.dendrogram(mushroomHierarchical, ax=axes[1, 2])
axes[1, 2].set_title('Mushrooms - Hierarchical Clustering')

# Plotting results for Wine dataset
axes[2, 0].scatter(wineDataXPCA[:, 0], wineDataXPCA[:, 1], c=wineDatabkmeans)
axes[2, 0].set_title('Wine - Bisecting k-Means')
axes[2, 1].scatter(wineDataXPCA[:, 0], wineDataXPCA[:, 1], c=wineDatakmeans)
axes[2, 1].set_title('Wine - Classic k-Means')
dendro = hierarchy.dendrogram(wineDataHierarchical, ax=axes[2, 2])
axes[2, 2].set_title('Wine - Hierarchical Clustering')

plt.show()