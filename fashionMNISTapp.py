import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st
import os

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 28, 28, 1).astype('float') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to build an enhanced CNN model
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Load or train the model
model_path = "enhanced_fashion_mnist_model.h5"
if os.path.exists(model_path):
    st.success("Model loaded from saved file.")
    model_fashion = load_model(model_path)
else:
    model_fashion = build_cnn_model()
    model_fashion.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    st.title("Fashion MNIST Model Training")
    st.write("Train a CNN model on the Fashion MNIST dataset.")

    if st.button("Start Training"):
        with st.spinner("Training in process..."):
            history = model_fashion.fit(X_train, y_train,
                                        validation_data=(X_test, y_test),
                                        epochs=20, batch_size=64)

        # Save model
        model_fashion.save(model_path)
        st.success("Model trained and saved successfully!")

        # Plot accuracy and loss
        fig1, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(history.history['accuracy'], label='Training Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(loc='upper left')

        ax[1].plot(history.history['loss'], label='Training Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend(loc='upper left')

        st.pyplot(fig1)

# Load model for prediction
if os.path.exists(model_path):
    model_fashion = load_model(model_path)

    # Confusion Matrix
    y_pred = model_fashion.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true_fashion = y_test.argmax(axis=1)

    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig2 = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        return fig2

    fig2 = plot_confusion_matrix(y_true_fashion, y_pred_classes, 'Enhanced CNN - Fashion MNIST')
    st.pyplot(fig2)

# File uploader and prediction
uploaded_file = st.file_uploader("Upload an image to classify", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = plt.imread(uploaded_file)
    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image, axis=2)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = tf.image.resize(image, (28, 28)).numpy()
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)

    st.image(image.reshape(28, 28), caption="Uploaded Image", width=150)

    if st.button("Predict"):
        predictions = model_fashion.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        st.write(f"**Predicted Class**: {class_names[predicted_class]}")
