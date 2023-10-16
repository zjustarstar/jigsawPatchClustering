import numpy as np
from typing import Union

def detect_neighbor(clu_result:np.ndarray,puzzle_size:Union[tuple, list],mode:str='remove') -> np.ndarray:
    """
    Detect and handle neighboring regions in clustering results.

    Args:
        clu_result (numpy.ndarray): A 1D NumPy array containing cluster labels for each patch.
        puzzle_size (tuple or list): A tuple or list specifying the number of puzzle pieces
                                      used during image segmentation, e.g., (2, 3).
        mode (str, optional): The mode for handling neighboring regions. 'remove' removes isolated regions,
                              and 'substitute' substitutes the label of isolated regions with their neighbors.
                              (default is 'remove')

    Returns:
        numpy.ndarray: A modified array containing cluster labels after handling neighboring regions.

    Example:
        To detect and handle neighboring regions in clustering results for a 2x3 grid of patches and remove
        isolated regions, you can call the function as follows:
        modified_labels = detect_neighbor(cluster_results, puzzle_size=(2, 3), mode='remove')
    """

    num_puzzle_h, num_puzzle_w = puzzle_size
    label_array = clu_result.reshape(num_puzzle_h,num_puzzle_w)
    new_label_array = np.zeros_like(label_array) + np.inf

    # 八个方向错位比较
    up_array = np.zeros_like(label_array) + np.inf
    up_array[:-1,:] = label_array[1:,:]
    new_label_array = np.where((label_array == up_array) , label_array,new_label_array)
    
    down_array = np.zeros_like(label_array) + np.inf
    down_array[1:,:] = label_array[:-1,:]
    new_label_array = np.where((label_array == down_array) , label_array,new_label_array)

    right_array = np.zeros_like(label_array) + np.inf
    right_array[:,1:] = label_array[:,:-1]
    new_label_array = np.where((label_array == right_array) , label_array,new_label_array)

    left_array = np.zeros_like(label_array) + np.inf
    left_array[:,:-1] = label_array[:,1:]
    new_label_array = np.where((label_array == left_array) , label_array,new_label_array)

    ur_array = np.zeros_like(label_array) + np.inf
    ur_array[:-1,1:] = label_array[1:,:-1]
    new_label_array = np.where((label_array == ur_array) , label_array,new_label_array)

    ul_array = np.zeros_like(label_array) + np.inf
    ul_array[:-1,:-1] = label_array[1:,1:]
    new_label_array = np.where((label_array == ul_array) , label_array,new_label_array)

    dr_array = np.zeros_like(label_array) + np.inf
    dr_array[1:,1:] = label_array[:-1,:-1]
    new_label_array = np.where((label_array == dr_array) , label_array,new_label_array)

    dl_array = np.zeros_like(label_array) + np.inf
    dl_array[1:,:-1] = label_array[:-1,1:]
    new_label_array = np.where((label_array == dl_array) , label_array,new_label_array)
    
    if np.max(new_label_array) == np.inf:
        if mode == 'remove':
            new_label_array[new_label_array == np.inf] = 1000 # means no label
        elif mode == 'substitute':
            # 选择最多邻居作为新标签
            nei_array = np.concatenate((up_array[new_label_array == np.inf].reshape(1,-1),
                            down_array[new_label_array == np.inf].reshape(1,-1),
                            right_array[new_label_array == np.inf].reshape(1,-1),
                            left_array[new_label_array == np.inf].reshape(1,-1),
                            ur_array[new_label_array == np.inf].reshape(1,-1),
                            ul_array[new_label_array == np.inf].reshape(1,-1),
                            dr_array[new_label_array == np.inf].reshape(1,-1),
                            dl_array[new_label_array == np.inf].reshape(1,-1),
                            ),axis=0)
            new_labels = []
            for i in range(nei_array.shape[-1]):
                v,c = np.unique(nei_array[:,i],return_counts=True)
                tmp_max = 0
                for j,k in zip(v,c): # 如果多个最大随机选择一个
                    if k > tmp_max and j != np.inf:
                        tmp_max = k
                        new_label = j
                new_labels.append(new_label)
            
            new_labels = np.array(new_labels)

            new_label_array[new_label_array == np.inf] = new_labels
            

    return new_label_array.reshape(-1,).astype(int)


def remove_class(labels:np.ndarray,M:int) -> np.ndarray:
    """
    Remove labels from clustering results if the count of patches assigned to the label is less than M.

    Args:
        labels (numpy.ndarray): A 1D NumPy array containing cluster labels for each patch.
        M (int): The minimum count of patches per cluster to keep the cluster.

    Returns:
        numpy.ndarray: A modified array containing cluster labels after removing labels with counts less than M.

    Example:
        To remove labels from clustering results 'cluster_labels' where clusters have fewer than 5 patches, you can call
        the function as follows:
        modified_labels = remove_class(cluster_labels, M=5)
    """

    values,counts = np.unique(labels,return_counts=True)
    values = np.where(counts >= M,values,1000)
    values = np.unique(values)
    new_labels = labels
    for i,l in enumerate(labels):
        if l not in values:
            new_labels[i] = 1000 # means no label
    return new_labels


def find_central_label(labels:np.ndarray,centers:np.ndarray,features:np.ndarray)->np.ndarray:
    # calculate distances of patches
    distances = np.zeros_like(labels).astype(float)

    for i, v in enumerate(labels):
        if v == 1000:
            continue
        center = centers[v]
        distances[i] = ((features[i] - center) ** 2).mean()  # ! calculate distance

    uni_label = np.unique(labels)

    # find min distance in every type of label and record corresponding index
    central_index = []
    for u in uni_label:
        if u == 1000:
            continue
        loc = np.where(labels == u)
        min_ind = np.argmin(distances[loc])
        central_index.append(loc[0][min_ind])

    return central_index


def remove_outlier(labels:np.ndarray,centers:np.ndarray,features:np.ndarray,thre:float) -> np.ndarray:
    """
    Remove outlier patches from clustering results based on distances to cluster centers.

    Args:
        labels (numpy.ndarray): A 1D NumPy array containing cluster labels for each patch.
        centers (numpy.ndarray): A 2D NumPy array containing cluster centers.
        features (numpy.ndarray): A 2D NumPy array containing features of each patch.
        thre (float): The threshold for considering patches as outliers.

    Returns:
        numpy.ndarray: A modified array containing cluster labels after removing outlier patches.

    Example:
        To remove outlier patches from clustering results 'cluster_labels' using cluster centers 'cluster_centers'
        and a threshold of 2.0, you can call the function as follows:
        modified_labels = remove_outlier(cluster_labels, cluster_centers, feature_matrix, thre=2.0)
    """
    
    # calculate distances of patches
    distances = np.zeros_like(labels).astype(float)
    
    for i,v in enumerate(labels):
        if v == 1000:
            continue
        center = centers[v]
        distances[i] = ((features[i] - center)**2).mean() #! calculate distance
    
    num_center = centers.shape[0]
    sigmas = np.zeros(shape=(2,num_center))
    uni_label = np.unique(labels)
    for u in uni_label:
        if u == 1000:
            continue
        mean = distances[labels == u].mean()
        std = distances[labels == u].std()
        sigmas[0,u] = mean + thre*std # maximum
        sigmas[1,u] = mean - thre*std # minimum
    
    new_labels = np.zeros_like(labels) + 1000
    for i,l in enumerate(labels):
        if l == 1000:
            continue
        if sigmas[1,l] < distances[i] <= sigmas[0,l]:
            new_labels[i] = labels[i]
        else:
            new_labels[i] = 1000

    return new_labels
