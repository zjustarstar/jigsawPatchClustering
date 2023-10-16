import numpy as np
import os
from skimage.io import imread
from sklearn.preprocessing import StandardScaler
import time
from typing import Union

from data import feature_extraction as fe
from data import label_postprocessing as lp
from data import feature_selection as fs
from model import kmeans_clustering as kc
from utils import visualization as vis

np.random.seed(42)

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
    features = np.concatenate((features,poi_feats),axis=-1)
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
    final_labels = lp.detect_neighbor(result,puzzle_size) # fix isolated patch (side effect:remove only-one labels).
    final_labels = lp.remove_class(final_labels,M)  # remove labels whose counts are less than M.
    final_labels = lp.remove_outlier(final_labels,centers,features,thre) # remove outlier according 3 sigma principle
    # final_labels = lp.detect_neighbor(final_labels,N)

    central_label_index = lp.find_central_label(final_labels,centers,features)

    return final_labels,central_label_index


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


if __name__ == '__main__':
    # Hyperparameters
    data_path = 'testimg'
    puzzle_size = (10, 10) # (num_puzzle_h:int,num_puzzle_w:int)
    K = 7   # K classes
    M = 4   # remove classes having less than M objects
    thre = 1.5  # threshold (3 sigma principle)

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img_list = [i for i in os.listdir(path=data_path) if i.endswith('.jpg')]
    non_border_patch_size = (puzzle_size[0]-2, puzzle_size[1]-2)
    for idx,img_name in enumerate(img_list):
        start_time = time.time()
        img = imread(os.path.join(data_path,img_name))

        patch_list, non_border_patch_list = image_segmentation(img,puzzle_size)
        non_border_final_labels, central_label_index = \
            image_clustering(non_border_patch_list, non_border_patch_size,K, M, thre,fs_method='vt')
        print(f'{img_name} is processing, {idx} : {time.time() - start_time}')

        final_full_labels = get_full_label(puzzle_size, non_border_final_labels)

        # save result
        fname, ext = os.path.splitext(img_name)
        name1 = fname + "_ori" + ext
        vis.plot_any(patch_list, puzzle_size, final_full_labels, image_name=name1, save_path='output')
        name2 = fname + "_lines" + ext
        vis.plot_by_lines(non_border_patch_list, non_border_final_labels, central_label_index, image_name=name2, K=K)
