import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import time
from typing import Union
from utils import visualization as vis
import clustering

np.random.seed(42)


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
    puzzle_size = (12, 12) # (num_puzzle_h:int,num_puzzle_w:int)
    K = 5   # K classes
    M = 4   # remove classes having less than M objects
    thre = 1.5  # threshold (3 sigma principle)

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img_list = [i for i in os.listdir(path=data_path) if i.endswith('.jpg')]
    non_border_patch_size = (puzzle_size[0]-2, puzzle_size[1]-2)
    for idx,img_name in enumerate(img_list):
        K = 5
        start_time = time.time()
        img = imread(os.path.join(data_path,img_name))

        # resize;
        img = resize(img, (img.shape[0]//4, img.shape[1]//4)) * 255
        img = img.astype(np.uint)

        patch_list, non_border_patch_list = clustering.image_segmentation(img,puzzle_size)
        non_border_final_labels, central_label_index, K = \
            clustering.image_clustering(non_border_patch_list, non_border_patch_size,K, M, thre,fs_method='vt')
        print(f'{img_name} is processing, {idx} : {time.time() - start_time}')

        final_full_labels = get_full_label(puzzle_size, non_border_final_labels)

        # save result
        fname, ext = os.path.splitext(img_name)
        name1 = fname + "_ori" + ext
        vis.plot_any(patch_list, puzzle_size, final_full_labels, image_name=name1, save_path='output')
        name2 = fname + "_lines" + ext
        vis.plot_by_lines(non_border_patch_list, non_border_final_labels, central_label_index, image_name=name2, K=K)
