import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow  


rf = Roboflow(api_key="DdyW2lm6d9UOewvrRfxk")  
project = rf.workspace("ysa-uewmt").project("tunbag")  
model = project.version(5).model  


image_path = r"C:\Users\ABC\Desktop\buburi.jpg"
predictions = model.predict(image_path).json()


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


for pred in predictions["predictions"]:
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    class_name = pred["class"]


    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)


    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
