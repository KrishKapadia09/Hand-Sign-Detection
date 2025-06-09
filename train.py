from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="C:/Users/use/OneDrive/Desktop/Image Processing/database works/hand sign/data.yaml", epochs=10 , imgsz=640, device=0)