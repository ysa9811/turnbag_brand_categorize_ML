from ml_image_data import mysql
from skimage.feature import hog, local_binary_pattern
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import albumentations as A
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# MobileNetV2 기반 특징 추출기
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# 데이터 증강
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, p=0.3),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.2),
])

def augment_image(image):
    return augmentations(image=image)["image"]

# 특징 추출 (HOG + LBP)
def extract_features(img):
    # HOG 특징 추출
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # LBP 특징 추출
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # 정규화

    # HOG + LBP 특징 벡터 결합
    features = np.hstack([hog_features, lbp_hist])
    return features

def extract_deep_features(img):
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    img = img / 255.0  # 정규화
    features = feature_extractor.predict(img)
    return features.flatten()

# 데이터 준비
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

            # 딥러닝 특징 추출 적용
            X.append(extract_deep_features(resized_img))
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 shape: {np.array(img).shape}")
            continue

    if len(X) == 0 or len(y) == 0:
        raise ValueError("데이터가 없습니다. 이미지 처리 오류를 확인하세요.")

    X = np.array(X)
    y = np.array(y).astype(np.int64)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 학습
def train_ml_model():
    img_all_data = mysql()
    X_train, X_test, y_train, y_test = prepare_ml_data(img_all_data)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

    # SVM 최적화
    model = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy:.4f}")

    return model

# 실행
ml_model = train_ml_model()
