import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO(r'C:\Users\user\Desktop\turnbag_brand_categorize_ML\runs\detect\train4\weights\best.pt')

folder_path = r"C:\Users\user\Desktop\API\test_image"

results_list = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)

        img = cv2.imread(image_path)

        results = model(img)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  
                confidence = float(box.conf[0])
                results_list.append((filename, class_id, confidence))
        
        predicted_image = results[0].plot()
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_image)
        plt.title(f"Predictions for {filename}")
        plt.axis('off')
        plt.show()

for filename, class_id, confidence in results_list:
    print(f"파일: {filename}, 클래스 ID: {class_id}, 신뢰도: {confidence:.2f}")
