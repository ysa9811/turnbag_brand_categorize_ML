import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("C:/Users/user/Desktop/yolo_training/tunbag_model/weights/best.pt")

input_folder = "C:/Users/user/Desktop/API/test_image"   

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file) 
    
    results = model(image_path)
    
    for result in results:
        img = result.plot() 

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")  
        plt.title(image_file)
        plt.show()
