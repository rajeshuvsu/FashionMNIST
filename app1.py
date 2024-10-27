import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st
from tensorflow.keras.models import load_model


(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

fig = plt.figure(figsize = (10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i],cmap='gray')
    plt.title(f"Label:{y_train[i]}")
    plt.axis('off')
plt.suptitle("Fashion MNIST sample image")

st.pyplot(fig)

X_train = X_train.reshape(-1,28,28,1).astype('float')/255
X_test = X_test.reshape(-1,28,28,1).astype('float')/255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

#funtion to build CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  
    model.add(Dense(256, activation='relu'))  
    model.add(Dropout(0.3)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax')) 
    return model

#compiling & training model
model_fashion = build_cnn_model()
model_fashion.compile   (optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

st.title("Fashion MNIST Model Training")
st.write("Train a CNN model on the Fashion MNIST dataset and view accuracy/loss plots and metrics.")

#if st.button("Strat Training"):
with st.spinner("Training in Process......"):
    h = model_fashion.fit(X_train,y_train,
                      validation_data = (X_test,y_test),
                      epochs=10,batch_size=64)

fig1, ax = plt.subplots(1, 2, figsize=(14, 5))


fig1, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(h.history['accuracy'], label='Training Accuracy')
ax[0].plot(h.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='upper left')

    
ax[1].plot(h.history['loss'], label='Training Loss')
ax[1].plot(h.history['val_loss'], label='Validation Loss')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='upper left')

    
st.pyplot(fig1)

metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(h.history['accuracy']) + 1)),
        "Training Accuracy": h.history['accuracy'],
        "Validation Accuracy": h.history['val_accuracy'],
        "Training Loss": h.history['loss'],
        "Validation Loss": h.history['val_loss']
    })

st.write("## Training Metrics")
st.table(metrics_df)

final_train_acc = h.history['accuracy'][-1] * 100
final_val_acc = h.history['val_accuracy'][-1] * 100
final_train_loss = h.history['loss'][-1]
final_val_loss = h.history['val_loss'][-1]

st.write("## Final Results:")
st.write(f"**Training Accuracy**: The training accuracy increased to {final_train_acc:.1f}% by the 10th epoch.")
st.write(f"**Validation Accuracy**: The validation accuracy stabilized around {final_val_acc:.1f}%, which is a decent result considering the complexity of the Fashion MNIST dataset.")
st.write(f"**Validation Loss**: The validation loss decreases steadily{final_val_loss:.1f}, although there's some fluctuation, indicating room for further improvement through hyperparameter tuning or additional techniques.")

model_fashion.save("fashion_mnist_model27.h5")
st.success("Model Trained and Saved Successfully!")

#if st.button("Load Mode"):
try:
    loaded_model = load_model('fashion_mnist_model27.h5')
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error("Model loading failed. Please train the model first.")
    loaded_model = None

    
st.write("## Generating the Confusion Matrices")
y_pred = loaded_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true_fashion = y_test if y_test.ndim == 1 else y_test.argmax(axis=1)

def plot_confusion_matrix(y_true,y_pred,title):
    cm = confusion_matrix(y_true,y_pred)
    fig2=plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Purples',cbar=False)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    return fig2


fig2 =plot_confusion_matrix(y_true_fashion,y_pred_classes,'Tuned MLP - Fashion MNIST')
st.pyplot(fig2)

st.write("## Conclusion")
st.write(
    "In this project, we successfully developed a Streamlit application for classifying images "
    "from the Fashion MNIST dataset using a Convolutional Neural Network (CNN). The application "
    "image preprocessing and visualization capabilities."
)

st.write("## Future Directions")
st.write(
    "While the current implementation is functional, there are several areas for potential improvement:"
)
future_directions = [
    "1. Enhanced Model Performance: Further hyperparameter tuning and data augmentation could improve accuracy.",
    "2. Model Interpretability: Techniques like Grad-CAM could help users understand the model's decisions.",
    "3. Batch Processing: Allowing users to upload multiple images for batch predictions.",
    "4. Real-Time Feedback Mechanism: Users can provide feedback on predictions to improve the model iteratively.",
    "5. Broader Dataset: Expanding the dataset with more diverse images could increase the model's applicability."
]
for direction in future_directions:
    st.write(direction)

