from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolov8s.pt")  

# 모델 학습
model.train(data=r'C:\Users\user\Desktop\tunbag.v5-sample.yolov8\data.yaml', epochs=100, patience=10, pretrained=True)
