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
        os.mkdir('./AugmentedDataset/')
    except:
        pass
    try:
        os.mkdir('./AugmentedDataset/train/')
        os.mkdir('./AugmentedDataset/train/images/')
        os.mkdir('./AugmentedDataset/train/labels/')
    except:
        pass
    for i in tqdm.tqdm(os.listdir('./Detection/train/images/')):
        img = cv2.imread(f'./Detection/train/images/{i}')
        with open(f'./Detection/train/labels/{i.split(".")[0]}.txt', "r", encoding="utf-8") as file:
            lines = file.read().splitlines()  # Tạo list, mỗi phần tử là 1 dòng
        img_arr = augment_image(img, 2024)
        count = 0
        for j in img_arr:
            cv2.imwrite(f'./AugmentedDataset/train/images/{i.split(".")[0]}_{count}.png', j)
            with open(f'./AugmentedDataset/train/labels/{i.split(".")[0]}_{count}.txt', "w", encoding="utf-8") as file:
                file.write("\n".join(lines))
            count += 1

    try:
        os.mkdir('./AugmentedDataset/val/')
        os.mkdir('./AugmentedDataset/val/images/')
        os.mkdir('./AugmentedDataset/val/labels/')
    except:
        pass
    for i in tqdm.tqdm(os.listdir('./Detection/val/images/')):
        img = cv2.imread(f'./Detection/val/images/{i}')
        with open(f'./Detection/val/labels/{i.split(".")[0]}.txt', "r", encoding="utf-8") as file:
            lines = file.read().splitlines()  # Tạo list, mỗi phần tử là 1 dòng
        cv2.imwrite(f'./AugmentedDataset/val/images/{i}', img)
        with open(f'./AugmentedDataset/val/labels/{i.split(".")[0]}.txt', "w", encoding="utf-8") as file:
            file.write("\n".join(lines))