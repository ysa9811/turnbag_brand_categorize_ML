from ml_image_data import mysql

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
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
            resized_img = cv2.resize(np.array(img, dtype=np.uint8), (128, 128))
            if len(resized_img.shape) == 2:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            X.append(resized_img)
            y.append(label)
        except Exception as e:
            print(f"이미지 처리 오류: {e}, 이미지 형태: {np.array(img).shape}")
            continue
    X = np.array(X) / 255.0
    y = np.array(y).astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_transfer_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    base_model.trainable = True
    fine_tune_at = 30
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
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

def train_model():
    img_all_data = mysql()
    X_train, X_test, y_train, y_test = prepare_data(img_all_data)
    num_classes = len(np.unique(y_train))
    model = create_transfer_model(num_classes)

    # 데이터 증강
    def augment_data(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        return image, label

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # 학습률 스케줄링
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001, first_decay_steps=10, t_mul=2.0, m_mul=0.9
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint('0123_1_best_model.keras', save_best_only=True, verbose=0),
        #EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # 테스트 결과
    test_loss, test_accuracy = model.evaluate(val_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 학습 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, history

model, history = train_model()
