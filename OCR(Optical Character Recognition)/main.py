import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image
import yaml
import os

# Load YOLO model
model = YOLO('yolov8n.pt')  # Start with a pre-trained model

# Load data config
with open('data.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)
field_names = data_cfg['names']

def train_yolo():
    model.train(data='data.yaml', epochs=50, imgsz=640)

def detect_fields(image_path):
    results = model(image_path)
    return results

def crop_and_ocr(image_path, results):
    img = cv2.imread(image_path)
    ocr_results = {}
    for r in results[0].boxes:
        cls = int(r.cls[0])
        label = field_names[cls]
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        crop = img[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop)
        ocr_results[label] = text.strip()
    return ocr_results

def visualize(image_path, results):
    img = cv2.imread(image_path)
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls = int(r.cls[0])
        label = field_names[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Detections', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    # 1. Train YOLO (uncomment if training)
    # train_yolo()

    # 2. Inference and OCR
    test_image = 'images/sample_passport.jpg'
    results = detect_fields(test_image)
    ocr_results = crop_and_ocr(test_image, results)
    print(ocr_results)

    # 3. Visualize
    visualize(test_image, results)
