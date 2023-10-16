import numpy as np # linear algebra
import matplotlib.pyplot as plt
import cv2
import skimage.feature
from skimage.feature.texture import graycomatrix, graycoprops
from typing import Union


def extract_color_features(img:np.ndarray) -> np.ndarray:
    """
    Extract color features from an input image using a multi-dimensional histogram.

    Args:
        img (numpy.ndarray): The input image from which color features will be extracted.
                            The image should be represented as a NumPy array.

    Returns:
        numpy.ndarray: A 1D NumPy array containing color features extracted from the input image.
                       The array represents a multi-dimensional histogram.

    Example:
        To extract color features from an image 'input_img', you can call the function as follows:
        color_features = extract_color_features(input_img)
    """
    
    # 多维直方图
    h,edges = np.histogramdd(img.reshape(-1,3),3,density=True,
                        range=[(0,255),(0,255),(0,255)])
    return h.flatten()


def extract_texture_features(image:np.ndarray) -> np.ndarray:
    """
    Extract texture features from an input image.

    Args:
        image (numpy.ndarray): The input image from which texture features will be extracted.
                               The image should be represented as a NumPy array.

    Returns:
        numpy.ndarray: A 1D NumPy array containing texture features extracted from the input image.
                       The array includes contrast, homogeneity, energy, and correlation features.

    Example:
        To extract texture features from an image 'input_img', you can call the function as follows:
        texture_features = extract_texture_features(input_img)
    """

    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # 计算LBP图像
    # lbp_image = skimage.feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    
    # # 计算LBP直方图
    # lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 10))
    
    # # 将直方图归一化
    # lbp_hist = lbp_hist.astype("float")
    # lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # 计算GCLM特征
    glcm = graycomatrix((gray_image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    texture_features_cont = graycoprops(glcm, 'contrast')
    texture_features_homo = graycoprops(glcm, 'homogeneity')
    texture_features_ener = graycoprops(glcm, 'energy')
    texture_features_corr = graycoprops(glcm, 'correlation')
    
    return np.concatenate([
                    # lbp_hist,
                    texture_features_cont.flatten(),
                    texture_features_homo.flatten(),
                    texture_features_ener.flatten(),
                    texture_features_corr.flatten()])

def extract_edge_features(image:np.ndarray) -> np.ndarray:
    """
    Extract edge features from an input image using Canny edge detection.

    Args:
        image (numpy.ndarray): The input image from which edge features will be extracted.
                               The image should be represented as a NumPy array.

    Returns:
        numpy.ndarray: A 1D NumPy array containing edge features extracted from the input image.
                       The array includes the count of edge pixels and the total contour area.

    Example:
        To extract edge features from an image 'input_img', you can call the function as follows:
        edge_features = extract_edge_features(input_img)
    """

    # 使用Canny边缘检测提取边缘特征
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)  # 调整阈值以控制边缘检测的灵敏度
    
    # 计算边缘像素的数量
    edge_pixel_count = np.sum(edges)

    # 计算边缘转角的数量
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = np.sum([cv2.contourArea(contour) for contour in contours])
    
    return np.array([edge_pixel_count,contour_area])

def extract_position_features(puzzle_size:Union[tuple, list]) -> np.ndarray:
    """
    Extract position features based on the puzzle size.

    Args:
        puzzle_size (tuple or list): A tuple or list specifying the number of puzzle pieces
                                      used during image segmentation, e.g., (2, 3).

    Returns:
        numpy.ndarray: A 2D NumPy array containing position features based on the puzzle size.
                       Each row of the array represents the position of a puzzle piece in the grid.

    Example:
        To extract position features for a 2x3 grid of puzzle pieces, you can call the function as follows:
        position_features = extract_position_features((2, 3))
    """

    num_puzzle_h, num_puzzle_w = puzzle_size
    row = np.arange(num_puzzle_h).reshape(-1,1).repeat(num_puzzle_w,1).reshape(-1,1)
    col = np.arange(num_puzzle_w).reshape(1,-1).repeat(num_puzzle_h,0).reshape(-1,1)

    return np.concatenate((row,col),axis=1)
