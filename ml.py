from ml_image_data import mysql
import requests
import pymysql
import requests
from io import BytesIO
import pandas as pd
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor

def prepare_data(img_all_data):
    X = []
    y = []

    for img, label in zip(img_all_data['image_rgb'], img_all_data['brand_name']):
        try:
            resized_img = cv2.resize(np.array(img, dtype=np.uint8), (224, 224))
            if len(resized_img.shape) == 2:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            X.append(resized_img)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 형태: {np.array(img).shape}")
            continue

    X = np.array(X)
    y = np.array(y)

    # 레이블 정규화 (1부터 시작하는 인덱스 문제 해결)
    unique_labels = np.unique(y)

    if np.any(unique_labels < 0) or np.any(unique_labels >= len(unique_labels)):
        raise ValueError(f"레이블 에러: {unique_labels}")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 데이터 정규화 (0~1 범위로 스케일링)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test

def create_transfer_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = True
    fine_tune_at = 100  # 학습할 레이어의 인덱스를 지정

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
        Dropout(0.3),
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
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        #EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3),
        ModelCheckpoint('0109_best_model.keras', save_best_only=True)  
    ]
    
    
    class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        validation_data=(X_test, y_test),
        epochs=40,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
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