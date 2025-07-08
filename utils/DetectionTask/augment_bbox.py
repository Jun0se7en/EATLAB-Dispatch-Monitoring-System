import cv2
import numpy as np
import random
import albumentations as A
from tqdm import tqdm
import os

def denormalize_bbox(bbox, img_width, img_height):
    """
    Chuyển đổi bounding box từ tọa độ chuẩn hóa về tọa độ pixel thực tế.
    bbox = (class, x_center, y_center, width, height)
    """
    cls, x, y, w, h = bbox
    x, y, w, h = int(x * img_width), int(y * img_height), int(w * img_width), int(h * img_height)
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x1 + w, y1 + h
    return cls, x1, y1, x2, y2

def normalize_bbox(cls, x1, y1, x2, y2, img_width, img_height):
    """
    Chuyển bounding box từ tọa độ pixel về tọa độ chuẩn hóa.
    """
    x = (x1 + x2) / 2 / img_width
    y = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return (cls, x, y, w, h)

def cut_and_paste_objects(image, bboxes):
    """
    Cắt các bounding boxes ra khỏi ảnh, trộn ngẫu nhiên, và dán lại vào các vị trí mới.
    """
    img_height, img_width, _ = image.shape
    cut_objects = []
    new_bboxes = []

    # Cắt vùng chứa bounding box và lưu lại
    for bbox in bboxes:
        cls, x1, y1, x2, y2 = denormalize_bbox(bbox, img_width, img_height)
        obj_patch = image[y1:y2, x1:x2].copy()
        cut_objects.append((obj_patch, cls, (x2 - x1), (y2 - y1)))  # Lưu vùng cắt + class + kích thước

        # Lấp đầy vùng đã cắt bằng một vùng ngẫu nhiên khác của ảnh
        while True:
            fill_x = random.randint(0, img_width - (x2 - x1))
            fill_y = random.randint(0, img_height - (y2 - y1))
            if not (fill_x <= x1 <= fill_x + (x2 - x1) and fill_y <= y1 <= fill_y + (y2 - y1)):  # Tránh chọn vùng trùng nhau
                break
        image[y1:y2, x1:x2] = image[fill_y:fill_y + (y2 - y1), fill_x:fill_x + (x2 - x1)]

    # Trộn vị trí và dán lại vào ảnh
    random.shuffle(cut_objects)
    for obj_patch, cls, w, h in cut_objects:
        new_x = random.randint(0, img_width - w)
        new_y = random.randint(0, img_height - h)
        image[new_y:new_y + h, new_x:new_x + w] = obj_patch
        new_bboxes.append(normalize_bbox(cls, new_x, new_y, new_x + w, new_y + h, img_width, img_height))

    return image, new_bboxes

def shear_bbox_level(image, bboxes, shear_factor_x=0.2, shear_factor_y=0.2):
    """
    Áp dụng biến dạng (Shear) lên từng bounding box riêng lẻ, không ảnh hưởng toàn bộ ảnh.
    :param image: Ảnh gốc (NumPy array)
    :param bboxes: Danh sách bounding boxes [(class, x_center, y_center, width, height)]
    :param shear_factor_x: Mức độ shear theo trục X
    :param shear_factor_y: Mức độ shear theo trục Y
    :return: Ảnh đã augment, bounding boxes mới
    """
    img_height, img_width, _ = image.shape
    new_bboxes = []

    for bbox in bboxes:
        cls, x_center, y_center, box_w, box_h = bbox

        # Chuyển từ normalized bbox sang pixel
        x1 = int((x_center - box_w / 2) * img_width)
        y1 = int((y_center - box_h / 2) * img_height)
        x2 = int((x_center + box_w / 2) * img_width)
        y2 = int((y_center + box_h / 2) * img_height)

        # Lấy object từ ảnh gốc
        object_patch = image[y1:y2, x1:x2].copy()

        # Tạo ma trận shear
        shear_x = random.uniform(-shear_factor_x, shear_factor_x)
        shear_y = random.uniform(-shear_factor_y, shear_factor_y)
        M = np.float32([
            [1, shear_x, 0],
            [shear_y, 1, 0]
        ])
        
        # Áp dụng shear lên object
        h, w = object_patch.shape[:2]
        sheared_patch = cv2.warpAffine(object_patch, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Dán object shear vào vị trí cũ
        image[y1:y2, x1:x2] = sheared_patch

        # Cập nhật lại bounding box sau shear
        new_x_center = (x1 + x2) / 2 / img_width
        new_y_center = (y1 + y2) / 2 / img_height
        new_box_w = (x2 - x1) / img_width
        new_box_h = (y2 - y1) / img_height

        new_bboxes.append((cls, new_x_center, new_y_center, new_box_w, new_box_h))

    return image, new_bboxes


def crop_bbox_level(image, bboxes, crop_ratio=(0.7, 0.9)):
    """
    Cắt crop bounding box ở level object, không phải toàn bộ ảnh.
    :param image: Ảnh gốc (NumPy array)
    :param bboxes: Danh sách bounding boxes [(class, x_center, y_center, width, height)]
    :param crop_ratio: Tỷ lệ cắt (giữ lại từ 70% - 90% của bounding box)
    :return: Ảnh đã augment, bounding boxes mới
    """
    img_height, img_width, _ = image.shape
    new_bboxes = []

    for bbox in bboxes:
        cls, x_center, y_center, box_w, box_h = bbox

        # Chuyển từ normalized bbox về pixel bbox
        x1 = int((x_center - box_w / 2) * img_width)
        y1 = int((y_center - box_h / 2) * img_height)
        x2 = int((x_center + box_w / 2) * img_width)
        y2 = int((y_center + box_h / 2) * img_height)

        # Xác định vùng crop trong bounding box
        crop_w = random.uniform(crop_ratio[0], crop_ratio[1]) * (x2 - x1)
        crop_h = random.uniform(crop_ratio[0], crop_ratio[1]) * (y2 - y1)

        crop_x1 = int(x1 + (x2 - x1 - crop_w) / 2)
        crop_y1 = int(y1 + (y2 - y1 - crop_h) / 2)
        crop_x2 = crop_x1 + int(crop_w)
        crop_y2 = crop_y1 + int(crop_h)

        # Cắt object ra và resize về kích thước ban đầu để giữ tính nhất quán
        object_patch = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        object_patch = cv2.resize(object_patch, ((x2 - x1), (y2 - y1)))

        # Dán object đã crop vào vị trí cũ
        image[y1:y2, x1:x2] = object_patch

        # Cập nhật bounding box mới về tọa độ chuẩn hóa
        new_x_center = (x1 + x2) / 2 / img_width
        new_y_center = (y1 + y2) / 2 / img_height
        new_box_w = (x2 - x1) / img_width
        new_box_h = (y2 - y1) / img_height

        new_bboxes.append((cls, new_x_center, new_y_center, new_box_w, new_box_h))

    return image, new_bboxes

### Augment Train Set ###
train_dir = './train/'
for i in tqdm(os.listdir(os.path.join(train_dir, 'images'))):
    s = i
    base, ext = os.path.splitext(s)
    image = cv2.imread(os.path.join(train_dir, 'images', i))
    with open(os.path.join(train_dir, 'labels', base + ".txt"), "r", encoding="utf-8") as file:
        lines = file.read().splitlines()  # Tạo list, mỗi phần tử là 1 dòng
    
    bboxes = []
    for j, line in enumerate(lines):
        tmp = list(map(float, line.split(" ")))
        tmp[0] = int(tmp[0])
        bboxes.append(tmp)
    
    # --- Thực hiện các Augmentations ---
    # 1. Cut & Paste
    cp_image, cp_bboxes = cut_and_paste_objects(image.copy(), bboxes)
    tmp_bboxes = []
    for i in cp_bboxes:
        tmp_bboxes.append(" ".join(list(map(str, i))))
    cp_bboxes = tmp_bboxes
    cv2.imwrite(os.path.join(train_dir, 'images', base + "_cp.jpg"), cp_image)
    with open(os.path.join(train_dir, 'labels', base + "_cp.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(cp_bboxes))
    # 2. Shear Augmentation
    sh_image, sh_bboxes = shear_bbox_level(image.copy(), bboxes, shear_factor_x=0.15, shear_factor_y=0.1)
    tmp_bboxes = []
    for i in sh_bboxes:
        tmp_bboxes.append(" ".join(list(map(str, i))))
    sh_bboxes = tmp_bboxes
    cv2.imwrite(os.path.join(train_dir, 'images', base + "_sh.jpg"), sh_image)
    with open(os.path.join(train_dir, 'labels', base + "_sh.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(sh_bboxes))
    # 3. Random Crop
    cr_image, cr_bboxes = crop_bbox_level(image.copy(), bboxes, crop_ratio=(0.7, 0.9))
    tmp_bboxes = []
    for i in cr_bboxes:
        tmp_bboxes.append(" ".join(list(map(str, i))))
    cr_bboxes = tmp_bboxes
    cv2.imwrite(os.path.join(train_dir, 'images', base + "_cr.jpg"), cr_image)
    with open(os.path.join(train_dir, 'labels', base + "_cr.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(cr_bboxes))
    
    
    
    
    

