from ml_image_data import mysql

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import cv2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(img_all_data):
    X = []
    y = []

    for img, label in zip(img_all_data['image_rgb'], img_all_data['brand_name']):
        try:
            resized_img = cv2.resize(np.array(img, dtype=np.uint8), (128, 128))
            if len(resized_img.shape) == 2:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            X.append(resized_img)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 형태: {np.array(img).shape}")
            continue

    X = np.array(X)
    y = np.array(y)

    unique_labels = np.unique(y)

    if np.any(unique_labels < 0) or np.any(unique_labels >= len(unique_labels)):
        raise ValueError(f"레이블 에러: {unique_labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return X_train, X_test, y_train, y_test

def create_transfer_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    
    base_model.trainable = True
    fine_tune_at = 50  

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),  
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),  
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model():
    img_all_data = mysql()
    print(img_all_data)
    print("=============================학습시작=============================")

    X_train, X_test, y_train, y_test = prepare_data(img_all_data)
    
    num_classes = len(np.unique(y_train))
    
    model = create_transfer_model(num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        #ReduceLROnPlateau(factor=0.2, patience=3),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint('0120_best_model.keras', save_best_only=True)
    ]
    
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))


    train_datagen = ImageDataGenerator(
        rotation_range=30,         # 이미지 회전 범위 (최대 30도)
        width_shift_range=0.2,     # 가로 방향 이동 범위
        height_shift_range=0.2,    # 세로 방향 이동 범위
        shear_range=0.2,           # 전단 변환 범위
        zoom_range=0.2,            # 확대/축소 범위
        horizontal_flip=True,      # 수평 반전
        fill_mode='nearest'        # 이미지 변환 시 채우는 방식
    )

    # 학습 데이터 증강기 생성
    train_generator = train_datagen.flow(
        X_train, y_train, 
        batch_size=32, 
        shuffle=True
    )

    # 검증 데이터는 증강하지 않음
    val_datagen = ImageDataGenerator()  # 증강 없이 그대로 사용
    val_generator = val_datagen.flow(
        X_test, y_test, 
        batch_size=32, 
        shuffle=False
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.int64),
        output_shapes=([None, 128, 128, 3], [None])
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_types=(tf.float32, tf.int64),
        output_shapes=([None, 128, 128, 3], [None])
    )

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 모델 학습
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=80,
        callbacks=callbacks,
        class_weight=class_weights
    )
        
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # 손실과 정확도를 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return model, history


model, history = train_model()