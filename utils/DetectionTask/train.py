from ultralytics import YOLO

model = YOLO("yolov8l.pt")  # Load a pretrained model

model.train(data="./AugmentedDataset/dataset.yaml",epochs=200, imgsz=640, batch=32, device=0,
            name="y8_dish_tray")  # Path to the dataset configuration file