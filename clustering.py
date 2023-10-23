import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Union

from data import feature_extraction as fe
from data import label_postprocessing as lp
from data import feature_selection as fs
from model import kmeans_clustering as kc


# np.random.seed(42)

def image_segmentation(img:np.ndarray,puzzle_size:Union[tuple, list]) -> list:
    """
    Perform image segmentation by dividing an input image into smaller patches.

    Args:
        img (numpy.ndarray): The input image to be segmented as a NumPy array.
        puzzle_size (tuple or list): A tuple or list specifying the number of puzzle pieces
                                      to divide the image horizontally and vertically, e.g., (2, 3).

    Returns:
        list of numpy.ndarray: A list containing segmented patches of the input image.
                               Each element in the list is a NumPy array representing a patch.

    Raises:
        AssertionError: If the dimensions of the image are not evenly divisible by the specified
                        puzzle_size, indicating an incorrect puzzle size.

    Example:
        To segment an image 'input_img' into a 2x3 grid of patches, you can call the function as follows:
        patches = image_segmentation(input_img, (2, 3))
    """
    # Image info
    h,w,_ = img.shape

    # Segmentation
    num_puzzle_h, num_puzzle_w = puzzle_size
    # assert h%num_puzzle_h == 0 and w%num_puzzle_w == 0, 'puzzle size wrong'
    patch_size = (h//num_puzzle_h,w//num_puzzle_w)
    patch_list = []

    non_border_patch = []
    for w_h in range(num_puzzle_h):
        for w_w in range(num_puzzle_w):
            patch_list.append(img[w_h*patch_size[0]:(w_h+1)*patch_size[0],w_w*patch_size[1]:(w_w+1)*patch_size[1],:])
            if 0<w_h<num_puzzle_h-1 and 0<w_w<num_puzzle_w-1:
                non_border_patch.append(img[w_h*patch_size[0]:(w_h+1)*patch_size[0],w_w*patch_size[1]:(w_w+1)*patch_size[1],:])

    return patch_list, non_border_patch


def image_clustering(patch_list:list,puzzle_size:Union[tuple, list],K:int=None,M:int=4,thre:float=3.0,fs_method: str=None ):
    """
    Perform image clustering on a list of image patches.

    Args:
        patch_list (list of numpy.ndarray): A list containing image patches represented as NumPy arrays.
        puzzle_size (tuple or list): A tuple or list specifying the number of puzzle pieces
                                      used during image segmentation.
        K (int, optional): The number of clusters to create. If None, the function will attempt to
                          find the optimal K value. (default is None)
        M (int, optional): Minimum count of patches per cluster to keep a cluster. (default is 4)
        thre (float, optional): The threshold for removing outliers using the 3-sigma principle.
                                (default is 3.0)
        fs_method (str, optional): The feature selection method to use ('vt' for VarianceThreshold).
                                  If None, no feature selection is performed. (default is None)

    Returns:
        list of int: A list of cluster labels assigned to each patch in the input 'patch_list'.
                     The labels are integers representing cluster assignments.
        central_label_index: central label index in every cluster type.

    Example:
        To cluster a list of image patches 'patch_list' with a specified K value of 5 and perform
        feature selection using VarianceThreshold, you can call the function as follows:
        cluster_labels = image_clustering(patch_list, puzzle_size=(2, 3), K=5, fs_method='vt')
    """

    # Extracting features
    # features = np.array([np.concatenate((fe.extract_color_features(i),
    #                                      fe.extract_texture_features(i), # LBP features are most time-consuming
    #                                      fe.extract_edge_features(i)))
    # for i in patch_list ])
    features = np.array([fe.extract_color_features(i) for i in patch_list])
    poi_feats = fe.extract_position_features(puzzle_size)
    # features = np.concatenate((features,poi_feats),axis=-1)
    scalar = StandardScaler()
    features = scalar.fit_transform(features)
    
    # Features selection
    if fs_method is not None:
        if fs_method == 'vt':
            fs.vt_selection(features)

    # Searching best k
    if K is None:
        # This function has not been accomplished, because the given K always equals to min_k or max_k according experiments. So, a predefined K may be more suitable.
        K = kc.search_k(min_k=4,max_k=8,features=features)

    # Clustering 
    km = kc.kmeans(K,features)
    result = km.predict(features)
    centers = km.cluster_centers_

    # Postprocessing
    #final_labels = lp.detect_neighbor(result,puzzle_size) # fix isolated patch (side effect:remove only-one labels).
    final_labels = result
    final_labels = lp.remove_class(final_labels,M)  # remove labels whose counts are less than M.
    # final_labels = lp.detect_neighbor(final_labels,N)
    final_labels = lp.remove_outlier(final_labels, centers, features,
                                     thre)  # remove outlier according 3 sigma principle

    central_label_index = lp.find_central_label(final_labels,centers,features)

    # re-cluster for big clustering.
    prek = K
    final_labels, central_label_index, K, centers = sub_cluster_resegment_bykmeans(puzzle_size, K, final_labels,
                                                                 features, central_label_index, centers)
    if not prek == K:
        final_labels = lp.remove_outlier(final_labels, centers, features,
                                         thre)  # remove outlier according 3 sigma principle


    return final_labels, central_label_index, K


# refill non-border-label back to full-label table.
def get_full_label(puzzle_size:Union[tuple, list], non_border_labels:list):
    hh, ww = puzzle_size[0], puzzle_size[1]
    final_labels = [1000] * hh * ww
    i = 0
    for r in range(hh):
        for c in range(ww):
            if 0 < r < hh - 1 and 0 < c < ww - 1:
                final_labels[c + r * ww] = non_border_labels[i]
                i = i + 1

    return final_labels


def sub_cluster_resegment_byfardist(puzzle_size, K, labels, features, center_index, centers):
    num_puzzle_h, num_puzzle_w = puzzle_size
    big_cluster_thre = num_puzzle_w * num_puzzle_h // 3 - 1
    uni_label = np.unique(labels)

    for u in uni_label:
        if u == 1000:
            continue
        loc = np.where(labels == u)
        loc = loc[0]

        if len(loc) > big_cluster_thre:
            # 计算每个特征离中心的距离,选择离其最远的点
            distances = np.zeros_like(loc).astype(float)
            for i, v in enumerate(loc):
                distances[i] = ((features[v] - centers[u]) ** 2).mean()  # ! calculate distance
            max_ind = np.argmax(distances)

            # 将距离最远的那个点，设为新的cluster center
            new_center_ind = loc[max_ind]
            nc_feature = features[new_center_ind]

            # 计算所有点离第二中心点的距离
            new_distance = np.zeros_like(loc).astype(float)
            for i, v in enumerate(loc):
                new_distance[i] = ((features[v] - nc_feature) ** 2).mean()  # ! calculate distance

            # 如果离第二中心点近，则归到第二中心点
            new_cluster_loc = np.where(distances - new_distance > 0)
            new_cluster_member_ind = loc[new_cluster_loc[0]]
            labels[new_cluster_member_ind] = K
            labels[new_center_ind] = K

            # 更新centers
            centers = np.append(centers, [nc_feature], axis=0)

            # 更新center_label
            center_index.append(new_center_ind)

            K = K + 1

    return labels, center_index, K, centers


# 如果某个cluster的block过多，则利用kmeans进行重新分割
def sub_cluster_resegment_bykmeans(puzzle_size, K, labels, features, center_index, centers):
    num_puzzle_h, num_puzzle_w = puzzle_size
    big_cluster_thre = num_puzzle_w * num_puzzle_h // 3 - 1
    uni_label = np.unique(labels)

    for u in uni_label:
        if u == 1000:
            continue
        loc = np.where(labels == u)
        loc = loc[0]

        if len(loc) > big_cluster_thre:
            sub_cluster_fea = features[loc]
            km = kc.kmeans(2, sub_cluster_fea)
            sub_labels = km.predict(sub_cluster_fea)
            sub_centers = km.cluster_centers_
            c_index = lp.find_central_label(sub_labels, sub_centers, sub_cluster_fea)
            central_label = loc[c_index]

            # 删除原有的centers，更新centers
            centers = np.delete(centers, u, axis=0)
            centers = np.append(centers, sub_centers, axis=0)

            # 更新labels:将数量少的一类label更新为K
            cluster0_label = loc[np.where(sub_labels == 0)]
            cluster1_label = loc[np.where(sub_labels == 1)]
            if len(cluster0_label) > len(cluster1_label):
                labels[cluster1_label] = K
            else:
                labels[cluster0_label] = K

            # 更新center_label
            for i in center_index:
                if i in loc:
                    center_index.remove(i)
                    break

            center_index.append(central_label[0])
            center_index.append(central_label[1])
            K = K + 1

    return labels, center_index, K, centers



