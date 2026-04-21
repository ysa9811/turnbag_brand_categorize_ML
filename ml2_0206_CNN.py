from ml_image_data import mysql
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 데이터 준비 함수
def prepare_ml_data(img_all_data):
    X, y = [], []
    
    for img, label in zip(img_all_data['image_rgb'], img_all_data['brand_name']):
        try:
            img = np.array(img, dtype=np.uint8)

            # 이미지 채널 확인 후 변환
            if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if len(img.shape) == 3 and img.shape[2] == 3:  # RGB -> Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 이미지가 1채널(그레이스케일)인지 확인
            if len(img.shape) == 2:
                gray_img = img  # 그대로 사용
            else:
                raise ValueError(f"예상치 못한 이미지 shape: {img.shape}")

            # 크기 조정 및 정규화
            resized_img = cv2.resize(gray_img, (128, 128)) / 255.0
            resized_img = np.expand_dims(resized_img, axis=-1)  # (128, 128, 1)로 변경
            
            X.append(resized_img)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 shape: {img.shape}")
            continue
    
    X = np.array(X)
    y = np.array(y).astype(np.int64)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Capsule Network 구조 예시 (간단한 버전)
def create_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # 과적합 방지
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model




# 모델 학습 함수
def train_ml_model():
    img_all_data = mysql()
    X_train, X_test, y_train, y_test = prepare_ml_data(img_all_data)

    # num_classes 설정
    num_classes = len(np.unique(y_train))

    # CNN 입력 형태 (128, 128, 1)로 변경
    model = create_cnn(input_shape=(128, 128, 1), num_classes=num_classes)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # 성능 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))  
    print(f"CapsNet Test Accuracy: {accuracy:.4f}")

# Grad-CAM 적용
def get_gradcam_heatmap(model, image, class_index, last_conv_layer_name):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

def apply_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

def visualize_gradcam(model, image, true_label, last_conv_layer='conv5_block16_concat'):
    image = np.array(image * 255, dtype=np.uint8)

    pred_probs = model.predict(np.expand_dims(image / 255.0, axis=0))[0]
    predicted_class = np.argmax(pred_probs)

    heatmap = get_gradcam_heatmap(model, image / 255.0, predicted_class, last_conv_layer)

    heatmap_overlay = apply_heatmap(image, heatmap)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Original (True: {true_label})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_overlay)
    plt.title(f"Overlay (Pred: {predicted_class})")
    plt.axis('off')

    plt.show()



# 모델 학습 실행
model, history, X_test, y_test = train_ml_model()

sample_index = np.random.randint(len(X_test))
sample_image = X_test[sample_index]
sample_label = y_test[sample_index]

visualize_gradcam(model, sample_image, sample_label)