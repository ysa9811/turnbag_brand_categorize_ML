from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO(r'C:\Users\user\Desktop\turnbag_brand_categorize_ML\runs\detect\train4\weights\best.pt')

image_path = r"C:\Users\user\Desktop\API\test_image\BALENCIAGA_bag1.jpg"
img = cv2.imread(image_path)

results = model(img)

# 결과 시각화
predicted_image = results[0].plot() 

# BGR → RGB 변환
predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

# 이미지 출력
plt.imshow(predicted_image)
plt.axis('off')  # 축 숨김
plt.show()
