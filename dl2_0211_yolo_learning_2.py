from ultralytics import YOLO
import shutil

model = YOLO("yolov8m.pt")  

results = model.train(
    data=r'C:\Users\kg098\Desktop\tunbag.v5-sample.yolov8\data.yaml',  
    epochs=300,            
    patience=0,             
    pretrained = True,
    verbose=True,             
    project="C:/Users/kg098/Desktop/yolo_training",  
    name="tunbag_model_2"     
)

best_model_path = "C:/Users/kg098/Desktop/yolo_training/tunbag_model_2/weights/best.pt"
destination_path = "C:/Users/kg098/Desktop/tunbag_trained_model_2.pt"

# best.pt 파일을 원하는 경로로 복사
shutil.copy(best_model_path, destination_path)

print(f"모델이 저장되었습니다: {destination_path}")
