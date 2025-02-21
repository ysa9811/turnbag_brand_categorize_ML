import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\Users\user\Desktop\brand_image\1111.jpg'
image = cv2.imread(image_path)

h, w, _ = image.shape
boxes = pytesseract.image_to_boxes(image)

for b in boxes.splitlines():
    b = b.split()
    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(image, (x, h - y), (x2, h - y2), (0, 255, 0), 2) 

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()

text = pytesseract.image_to_string(image)
print("인식된 텍스트:")
print(text)
