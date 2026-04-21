import tensorflow as tf
import numpy as np
import cv2
import os

# 클래스 매핑
"""
answer = {
    0: 'BALENCIAGA', 1: 'BOTTEGA VENETA', 2: 'BURBERRY', 3: 'CELINE', 4: 'Citizen',
    5: 'DIOR', 6: 'Fendi', 7: 'GUCCI', 8: 'Hamilton', 9: 'HERMES', 10: 'Longines',
    11: 'Louis Vuitton', 12: 'MAISON246', 13: 'Michael Kors', 14: 'MONCLER',
    15: 'Rolex', 16: 'ROMANSON', 17: 'THOM BROWNE', 18: 'SALVATORE FERRAGAMO',
    19: 'PRADA', 20: 'Chanel'
}
"""
answer = {
    0: 'BALENCIAGA', 1: 'BOTTEGA VENETA', 2: 'BURBERRY', 3: 'CELINE', 4: 'DIOR',
    5: 'GUCCI', 6: 'HERMES', 7: 'Louis Vuitton', 8: 'MONCLER', 9: 'THOM BROWNE', 10: 'SALVATORE FERRAGAMO',
    11: 'PRADA', 12: 'Chanel'
}
# 모델 로드
model = tf.keras.models.load_model('0210_best_model(92).keras')

# 이미지 전처리 함수
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        img_resized = cv2.resize(img, (128, 128))  # 🔥 224 → 128로 변경
        img_normalized = img_resized / 255.0
        img_final = np.expand_dims(img_normalized, axis=0).astype(np.float32)  # 🔥 float32 유지
        
        return img_final
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return None

# 개별 이미지 예측 함수
def predict_image(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    
    if preprocessed_img is None:
        return None  # 이미지 불러오기 실패 시 None 반환
    
    predictions = model.predict(preprocessed_img)
    predicted_label_idx = np.argmax(predictions, axis=1)[0]  # 🔥 가장 높은 확률의 클래스 선택
    
    return predicted_label_idx

# 폴더 내 모든 이미지 처리
def predict_folder(model, folder_path):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 🔥 이미지 파일 확장자 필터링
            image_path = os.path.join(folder_path, filename)
            predicted_label_idx = predict_image(model, image_path)
            
            if predicted_label_idx is not None:
                predicted_label = answer.get(predicted_label_idx, "Unknown")
                results.append((filename, predicted_label))
    
    return results

# 실행
folder_path = r"C:\Users\user\Desktop\API\test_image"
predictions = predict_folder(model, folder_path)

# 결과 출력
for filename, label in predictions:
    print(f"파일: {filename}, 예측 결과: {label}")
