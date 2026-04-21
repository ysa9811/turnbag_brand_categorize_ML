from ml_image_data import mysql

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import cv2

# Mixed precision 사용
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def prepare_data(img_all_data):
    X, y = [], []
    for img, label in zip(img_all_data['image_rgb'], img_all_data['brand_name']):
        try:
            # 이미지를 uint8 타입의 numpy array로 변환 후 리사이즈
            img_array = np.array(img, dtype=np.uint8)
            resized_img = cv2.resize(img_array, (128, 128))
            
            # 이미지의 채널 수에 따라 변환
            if len(resized_img.shape) == 2:
                # Grayscale image: (128,128) -> (128,128,3)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            elif resized_img.shape[-1] == 4:
                # RGBA 이미지: (128,128,4) -> (128,128,3)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)
            # 만약 채널 수가 3가 아니라면 (예: BGR), 필요에 따라 변환 처리
            elif resized_img.shape[-1] != 3:
                # 예시: BGR -> RGB (OpenCV 기본값은 BGR)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            X.append(resized_img)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 형태: {np.array(img).shape}")
            continue
    X = np.array(X) / 255.0  # 모든 이미지가 (128,128,3)로 동일한 shape이어야 함
    y = np.array(y).astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)  # 좌우 반전은 유지 (로고 대칭 변화 가능)
    image = tf.image.random_brightness(image, 0.1)  # 밝기 변화를 줄여서 색상 보존
    image = tf.image.random_contrast(image, 0.9, 1.1)  # 콘트라스트 변화 범위 축소
    image = tf.image.random_hue(image, 0.05)  # 색상 변화 범위 축소
    image = tf.image.random_saturation(image, 0.9, 1.1)  # 채도 변화 축소
    image = tf.image.resize_with_crop_or_pad(image, 140, 140)  # 패딩 추가 (잘림 방지)
    image = tf.image.random_crop(image, size=(128, 128, 3))  # 원래 크기로 크롭 (랜덤 Crop 제거 가능)
    return image, label


def create_transfer_model(num_classes):
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    base_model.trainable = True  # DenseNet121의 가중치를 학습 가능하도록 설정
    fine_tune_at = 80  # Fine-tuning 범위 확장

    # 초기 레이어 동결
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 모델 정의
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])

    return model

def train_model():
    img_all_data = mysql()
    X_train, X_test, y_train, y_test = prepare_data(img_all_data)
    num_classes = len(np.unique(y_train))

    model = create_transfer_model(num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    initial_lr = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps=10000, alpha=0.0001)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint('0204_best_model.keras', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=callbacks,
        class_weight=class_weights
    )

    test_loss, test_accuracy = model.evaluate(val_dataset)
    print(f"Test Accuracy with DenseNet121: {test_accuracy:.4f}")

    return model, history, X_test, y_test

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

# 모델 학습 및 Grad-CAM 테스트
model, history, X_test, y_test = train_model()

sample_index = np.random.randint(len(X_test))
sample_image = X_test[sample_index]
sample_label = y_test[sample_index]

visualize_gradcam(model, sample_image, sample_label)