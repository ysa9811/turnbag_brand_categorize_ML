from ml_image_data import mysql
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
from xgboost import XGBClassifier 

# 데이터 준비 함수
def prepare_ml_data(img_all_data):
    X, y = [], []
    
    for img, label in zip(img_all_data['image_rgb'], img_all_data['brand_name']):
        try:
            # 이미지 전처리 (128x128 변환)
            resized_img = cv2.resize(np.array(img, dtype=np.uint8), (128, 128))

            # RGBA(4채널) -> RGB(3채널) 변환 후 흑백 변환
            if len(resized_img.shape) == 3:
                if resized_img.shape[2] == 4:  # RGBA 이미지 처리
                    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGBA2RGB)
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = resized_img  # 이미 흑백이면 그대로 사용

            # HOG 특징 추출
            features = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 shape: {np.array(img).shape}")
            continue
    
    X = np.array(X)
    y = np.array(y).astype(np.int64)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# 모델 학습 함수
def train_ml_model():
    img_all_data = mysql()
    X_train, X_test, y_train, y_test = prepare_ml_data(img_all_data)

    model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 성능 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy:.4f}")

    return model

# 모델 학습 실행
ml_model = train_ml_model()
