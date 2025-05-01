1. Setup and Installation
Let's start by installing the required libraries that will allow us to process the CT scan images, build the deep learning model, and create a user-friendly web interface for interaction. We’ll use libraries like TensorFlow, OpenCV, Scikit-learn, Matplotlib, and Gradio for this task.

You can install these by running:

bash
Copy
Edit
!pip install tensorflow opencv-python scikit-learn matplotlib gradio
2. Mount Google Drive
To access your CT scan image and mask datasets stored on Google Drive, we first need to mount the drive. This allows us to read the files directly from Google Drive into our environment.

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
3. Load and Preprocess Data
The next step involves loading and preprocessing the data. We're assuming that your CT scan images and their corresponding masks are stored in two directories: one for training images (frames_train) and another for test images (frames_test). The masks are grayscale images where the infected areas are marked.

python
Copy
Edit
import os
import cv2
import numpy as np

IMG_SIZE = 128  # Resize images to this size

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

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize image
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))  # Resize mask
        mask = np.expand_dims((mask > 0).astype(np.uint8), axis=-1)  # Convert to binary mask

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load training and testing data
X_train, y_train = load_data('/content/drive/MyDrive/COVID/frames_train/', '/content/drive/MyDrive/COVID/masks_train/')
X_test, y_test = load_data('/content/drive/MyDrive/COVID2/frames_test/', '/content/drive/MyDrive/COVID2/masks_test/')
4. Model: Hybrid U-Net + ResNet Architecture
Now, let’s build our deep learning model. We are combining U-Net and ResNet50 to take advantage of both models’ strengths: U-Net for segmentation tasks and ResNet50 for feature extraction. The model will be used to predict the segmentation mask for each CT scan.

python
Copy
Edit
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet_resnet_hybrid(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Use pre-trained ResNet50 as the base model for feature extraction
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = False  # Freeze the ResNet50 layers

    # Skip connections from ResNet layers
    skips = [
        base_model.get_layer("conv1_relu").output,
        base_model.get_layer("conv2_block3_out").output,
        base_model.get_layer("conv3_block4_out").output,
        base_model.get_layer("conv4_block6_out").output
    ]
    
    # Bottleneck layer
    x = base_model.get_layer("conv5_block3_out").output

    # Decoder part with skip connections
    for skip in reversed(skips):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # Final convolution layers
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    # Output layer with sigmoid activation for binary mask prediction
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

# Build and compile the model
model = build_unet_resnet_hybrid()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=6, batch_size=8, validation_data=(X_test, y_test))
5. Model Evaluation
Once the model is trained, we evaluate its performance on the test dataset. We use the accuracy metric to assess how well the model is predicting the segmentation mask.

python
Copy
Edit
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(np.uint8)  # Convert to binary mask

# Calculate accuracy
accuracy = accuracy_score(y_test.flatten(), y_pred_binary.flatten())
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
6. Visualizing Predictions
Now, let’s visualize the model’s predictions. We will show the original CT scan image, the true mask, and the predicted mask side by side for comparison.

python
Copy
Edit
import matplotlib.pyplot as plt

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

# Visualize the first two predictions
plot_prediction(0)
plot_prediction(1)
7. Gradio User Interface
Finally, let’s create a simple and interactive web interface using Gradio where users can upload a CT scan image, and the model will predict and display the segmentation mask.

python
Copy
Edit
import gradio as gr

def predict_segmentation(image):
    """
    This function takes an uploaded image, preprocesses it, 
    and predicts the segmentation mask using the trained model.
    """
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    
    # Predict the segmentation mask
    mask_pred = model.predict(img_resized)
    mask_pred_binary = (mask_pred > 0.5).astype(np.uint8)
    
    # Return the predicted mask
    return mask_pred_binary[0].squeeze()

# Create Gradio interface
iface = gr.Interface(fn=predict_segmentation, 
                     inputs=gr.inputs.Image(type="numpy", shape=(IMG_SIZE, IMG_SIZE, 3)),
                     outputs=gr.outputs.Image(type="numpy", label="Predicted Mask"),
                     live=True)

# Launch the interface
iface.launch()
How It Works:
User Uploads a CT Scan Image: The user can upload a CT scan image via the Gradio interface.

Model Predicts the Mask: The model processes the image, predicts the segmentation mask, and shows it on the interface.

Output: The predicted mask is displayed to the user as an image, helping them visualize which areas of the lung are infected with COVID-19.

8. Running the Application
Once you run the complete code in your notebook, Gradio will open up a web-based interface where you can:

Upload a CT scan image.

See the predicted segmentation mask generated by the model.
