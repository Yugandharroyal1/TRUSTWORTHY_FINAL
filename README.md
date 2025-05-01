# Install dependencies
!pip install tensorflow opencv-python scikit-learn matplotlib gradio

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import required libraries
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    InceptionResNetV2, DenseNet121, ResNet50, VGG19,
    MobileNetV2, InceptionV3
)

# Set image size
IMG_SIZE = 128

# Function to load dataset
def load_data(image_dir, mask_dir):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.expand_dims((mask > 0).astype(np.uint8), axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load datasets
X_train, y_train = load_data('/content/drive/MyDrive/COVID/frames_train/', '/content/drive/MyDrive/COVID/masks_train/')
X_test, y_test = load_data('/content/drive/MyDrive/COVID2/frames_test/', '/content/drive/MyDrive/COVID2/masks_test/')

# Model builders
def build_model(base_model_class, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_unet_resnet_hybrid(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = False

    skips = [
        base_model.get_layer("conv1_relu").output,
        base_model.get_layer("conv2_block3_out").output,
        base_model.get_layer("conv3_block4_out").output,
        base_model.get_layer("conv4_block6_out").output
    ]
    x = base_model.get_layer("conv5_block3_out").output

    for skip in reversed(skips):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

# Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, is_segmentation=False):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=6, batch_size=8, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    accuracy = accuracy_score(y_test.flatten(), y_pred_binary.flatten())
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
    return y_pred_binary

# Run all models
models_to_run = {
    "InceptionResNetV2": lambda: build_model(InceptionResNetV2),
    "DenseNet": lambda: build_model(DenseNet121),
    "ResNet": lambda: build_model(ResNet50),
    "VGG19": lambda: build_model(VGG19),
    "MobileNet": lambda: build_model(MobileNetV2),
    "InceptionV3": lambda: build_model(InceptionV3),
    "UNet+ResNet": build_unet_resnet_hybrid
}

for name, model_fn in models_to_run.items():
    print(f"\nRunning {name}...\n")
    model = model_fn()
    is_segmentation = (name == "UNet+ResNet")
    y_pred_binary = train_and_evaluate(model, X_train, y_train, X_test, y_test, is_segmentation)

    if is_segmentation:
        # Visualize predictions
        def plot_prediction(index):
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(X_test[index])
            plt.title("Input Image")
            plt.subplot(1, 3, 2)
            plt.imshow(y_test[index].squeeze(), cmap='gray')
            plt.title("True Mask")
            plt.subplot(1, 3, 3)
            plt.imshow(y_pred_binary[index].squeeze(), cmap='gray')
            plt.title("Predicted Mask")
            plt.show()

        plot_prediction(0)
        plot_prediction(1)
