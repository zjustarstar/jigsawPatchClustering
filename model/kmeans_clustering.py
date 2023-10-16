from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

def kmeans(K:int,features:np.ndarray) -> KMeans:
    """
    Perform K-means clustering on the given features.

    Args:
        K (int): The number of clusters to create.
        features (numpy.ndarray): A 2D NumPy array containing feature vectors for clustering.

    Returns:
        sklearn.cluster.KMeans: A KMeans object representing the K-means clustering model.

    Example:
        To perform K-means clustering with K=5 on a feature matrix 'features', you can call the function as follows:
        kmeans_model = kmeans(K=5, features=features)
    """

    km=KMeans(n_clusters=K,
             init='k-means++',
             n_init=10,
             max_iter=300,
             random_state=42)
    km.fit(features)
    return km

def search_k(min_k:int,max_k:int,features:np.ndarray) -> int: 
    """
    Search for the optimal number of clusters (K) using various clustering evaluation metrics.

    Args:
        min_k (int): The minimum number of clusters to consider during the search.
        max_k (int): The maximum number of clusters to consider during the search.
        features (numpy.ndarray): A 2D NumPy array containing feature vectors for clustering.

    Returns:
        int: The recommended optimal number of clusters (K) based on clustering evaluation metrics.

    Example:
        To search for the optimal number of clusters between 2 and 10 using clustering evaluation metrics for a feature matrix
        'features', you can call the function as follows:
        optimal_k = search_k(min_k=2, max_k=10, features=features)

    Note:
        This function has not been accomplished, because the given K always equals to min_k or max_k according experiments. So a predefined K may be more suitable.
    """

    distortions = []
    SSE = []
    SC = []
    CH = []
    for k in range(min_k,max_k+1):
        km=kmeans(k,features)
        y1=km.predict(features)
        sc=metrics.silhouette_score(features,y1)
        SC.append(sc)
        ch=metrics.calinski_harabasz_score(features,y1)
        CH.append(ch)
        distortions.append(km.inertia_)
        SSE.append(km.inertia_)
    best_idx = np.array([distortions.index(min(distortions))+min_k,SSE.index(min(SSE))+min_k,SC.index(max(SC))+min_k,CH.index(max(CH))+min_k])
    idx,counts = np.unique(best_idx,return_counts=True)

    return idx[np.argmax(counts)]


