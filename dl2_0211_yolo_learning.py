from ultralytics import YOLO
import shutil

# YOLO 모델 로드 (더 강력한 모델 사용)
model = YOLO("yolov8m.pt")  

# 모델 학습 (최적화 설정)
results = model.train(
    data=r'C:\Users\kg098\Desktop\tunbag.v5-sample.yolov8\data.yaml',  
    epochs=300,            
    patience=0,          
    batch=-1,                
    imgsz=640,            
    optimizer='AdamW',      
    lr0=0.001,                
    momentum=0.937,          
    weight_decay=0.0005,     
    hsv_h=0.015,            
    hsv_s=0.7,                
    hsv_v=0.4,               
    degrees=5,              
    translate=0.1,          
    scale=0.5,             
    flipud=0.1,              
    fliplr=0.5,              
    mosaic=0.8,               
    mixup=0.1,             
    pretrained=True,        
    workers=4,               
    verbose=True,           
    project="C:/Users/kg098/Desktop/yolo_training",  
    name="tunbag_model"     
)

best_model_path = "C:/Users/kg098/Desktop/yolo_training/tunbag_model/weights/best.pt"
destination_path = "C:/Users/kg098/Desktop/tunbag_trained_model.pt"

# best.pt 파일을 원하는 경로로 복사
shutil.copy(best_model_path, destination_path)

print(f"모델이 저장되었습니다: {destination_path}")
