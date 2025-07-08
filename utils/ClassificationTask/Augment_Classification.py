import tensorflow as tf
import os
import cv2
import numpy as np
import random
import tqdm

def augment_image(image, seed):
    image_arr = []
    image_arr.append(image)
    # Brightness Augment
    for i in range(5):
        bright_image = tf.image.random_brightness(image, 0.4, seed)
        image_arr.append(bright_image)
    # Hue Augment
    for i in range(5):
        hue_image = tf.image.random_hue(image, 0.5, seed)
        image_arr.append(hue_image)
    # # Contrast Augment
    for i in range(5):
        contrast_image = tf.image.random_contrast(image, 0.15, seed)
        image_arr.append(contrast_image)
    
    return np.array(image_arr)

if __name__ == "__main__":
    try:
        os.mkdir('./AugmentedDatasetClassification/')
    except:
        pass
    try:
        os.mkdir('./AugmentedDatasetClassification/dish/')
        os.mkdir('./AugmentedDatasetClassification/dish/empty/')
        os.mkdir('./AugmentedDatasetClassification/dish/kakigori/')
        os.mkdir('./AugmentedDatasetClassification/dish/not_empty/')
        os.mkdir('./AugmentedDatasetClassification/tray/')
        os.mkdir('./AugmentedDatasetClassification/tray/empty/')
        os.mkdir('./AugmentedDatasetClassification/tray/kakigori/')
        os.mkdir('./AugmentedDatasetClassification/tray/not_empty/')
    except:
        pass
    for i in tqdm.tqdm(os.listdir('./Classification/')):
        for j in tqdm.tqdm(os.listdir(f'./Classification/{i}/')):
            for k in tqdm.tqdm(os.listdir(f'./Classification/{i}/{j}/')):
                # print(f'./Classification/{i}/{j}/{k}')
                img = cv2.imread(f'./Classification/{i}/{j}/{k}')
                # print(img.shape)
                img_arr = augment_image(img, 2024)
                count = 0
                for o in img_arr:
                    cv2.imwrite(f'./AugmentedDatasetClassification/{i}/{j}/{k.split(".")[0]}_{count}.png', o)
                    count += 1