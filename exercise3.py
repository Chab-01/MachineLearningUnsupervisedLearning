import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def sammon(X, iter, errorThreshold, learningRate):
    n, p = X.shape
    Y = np.random.rand(n, 2)
    epsilon = 1e-4  # Small constant to handle division by zero

    # Compute the stress E of Y
    def compute_stress():
        distX = np.sqrt(np.sum(np.square(X[:, np.newaxis, :] - X), axis=2))
        distY = np.sqrt(np.sum(np.square(Y[:, np.newaxis, :] - Y), axis=2))
        return np.sum(np.abs(distX - distY) / (distX + epsilon))  # We add epsilon to make sure we don't divide by zero even if distX is 0

    stress = compute_stress()

    i = 0
    while stress > errorThreshold and i < iter:
        for j in range(n):
            delta = np.zeros(2)
            for k in range(n):
                if k != j:
                    distX_jk = np.sqrt(np.sum(np.square(X[j] - X[k])) + epsilon)
                    distY_jk = np.sqrt(np.sum(np.square(Y[j] - Y[k])) + epsilon)
                    gradient = (distX_jk - distY_jk) / (distY_jk * (distX_jk + epsilon))
                    delta += gradient * (Y[j] - Y[k])
            Y[j] -= learningRate * delta

        newStress = compute_stress()
        if newStress >= stress:
            break

        stress = newStress
        i += 1

    return Y

techniques = ["Sammon Mapping", "PCA", "t-SNE"]
for technique in techniques:
    if technique == "Sammon Mapping":
        resultsCancer = sammon(breastCancerX, iter=2, errorThreshold=1e-4, learningRate=0.1)
        resultsDiabetes = sammon(mushroomX, iter=2, errorThreshold=1e-4, learningRate=0.1)
        resultsTrial = sammon(wineDataX, iter=2, errorThreshold=1e-4, learningRate=0.1)
        plt.subplot(3,3,1)
        plt.scatter(resultsCancer[:, 0], resultsCancer[:, 1], c=breastCancerLabels)
        plt.subplot(3,3,2)
        plt.scatter(resultsDiabetes[:, 0], resultsDiabetes[:, 1], c=mushroomLabels)
        plt.subplot(3,3,3)
        plt.ylabel("Sammon")
        plt.scatter(resultsTrial[:, 0], resultsTrial[:, 1], c=wineLabels)
    elif technique == "PCA":
        pca = PCA(n_components=2)
        resultsCancer = pca.fit_transform(breastCancerX)
        resultsDiabetes = pca.fit_transform(mushroomX)
        resultsTrial = pca.fit_transform(wineDataX)
        plt.subplot(3,3,4)
        plt.scatter(resultsCancer[:, 0], resultsCancer[:, 1], c=breastCancerLabels)
        plt.subplot(3,3,5)
        plt.scatter(resultsDiabetes[:, 0], resultsDiabetes[:, 1], c=mushroomLabels)
        plt.subplot(3,3,6)
        plt.ylabel("PCA")
        plt.scatter(resultsTrial[:, 0], resultsTrial[:, 1], c=wineLabels)
    elif technique == "t-SNE":
        tsne = TSNE(n_components=2)
        resultsCancer = tsne.fit_transform(breastCancerX)
        resultsDiabetes = tsne.fit_transform(mushroomX)
        resultsTrial = tsne.fit_transform(wineDataX)
        plt.subplot(3,3,7)
        plt.scatter(resultsCancer[:, 0], resultsCancer[:, 1], c=breastCancerLabels)
        plt.subplot(3,3,8)
        plt.scatter(resultsDiabetes[:, 0], resultsDiabetes[:, 1], c=mushroomLabels)
        plt.subplot(3,3,9)
        plt.ylabel("T-SNE")
        plt.scatter(resultsTrial[:, 0], resultsTrial[:, 1], c=wineLabels)
    plt.xlabel(techniques)

plt.tight_layout()
plt.show()
