"""
AI-Generated vs. Real Human Face Classifier
Author: Sümeyye Bakırdal
Description: A Convolutional Neural Network (CNN) built with TensorFlow/Keras 
to distinguish between real human faces and AI-generated faces.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import pathlib

# --- Image Loading Function ---
def load_images_from_folder(folder, label):
    """
    Loads images from a directory, converts to RGB, 
    resizes to 224x224, and assigns a label.
    """
    images = []
    labels = [] 
    if not os.path.exists(folder):
        print(f"Warning: Folder not found -> {folder}")
        return images, labels

    print(f"Loading images from: {folder}")
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize for the CNN model input
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
    return images, labels

# --- Path Configurations ---
# Note: Update these paths or use a relative 'data/' folder for GitHub portability
human_path = r"data/human"
ai_path = r"data/ai"

# Load Human faces (Label: 0) and AI faces (Label: 1)
human_images, human_labels = load_images_from_folder(human_path, 0)
ai_images, ai_labels = load_images_from_folder(ai_path, 1)

# --- Data Preprocessing ---
print("Preprocessing dataset...")
# Convert to numpy arrays and normalize pixel values to [0, 1]
human_images = np.array(human_images, dtype=np.float32) / 255.0
ai_images = np.array(ai_images, dtype=np.float32) / 255.0
human_labels = np.array(human_labels)
ai_labels = np.array(ai_labels)

# Concatenate all data
X = np.concatenate((human_images, ai_images), axis=0)
y = np.concatenate((human_labels, ai_labels), axis=0).astype("float32")

# --- Dataset Splitting ---
# 70% Training, 30% Remaining (for Val/Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split remaining 30% into Validation and Test (15% each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- CNN Model Architecture ---
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fully Connected Layer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularization to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary Output (0 or 1)
])

# --- Compile Model ---
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# --- Model Training ---
print("Starting model training...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=10, 
    batch_size=16
)

# Display model summary
model.summary()

# --- Evaluation & Metrics ---
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=['Human', 'AI']))

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\nOverall Model Accuracy: {accuracy:.4f}")

# --- Visualization: Training Performance ---
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --- Visualization: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- Visualization: Sample Predictions ---
def show_sample_results(model, images, true_labels, threshold=0.5, num_images=10):
    """Displays sample images with their true and predicted labels."""
    indices = np.random.choice(len(images), num_images, replace=False)
    sample_images = images[indices]
    sample_true = true_labels[indices]

    predictions = model.predict(sample_images)
    predicted_labels = (predictions >= threshold).astype(int).flatten()

    plt.figure(figsize=(20, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i])
        plt.axis("off")

        true_text = "Human" if sample_true[i] == 0 else "AI"
        pred_text = "Human" if predicted_labels[i] == 0 else "AI"
        color = "green" if predicted_labels[i] == sample_true[i] else "red"
        
        plt.title(f"True: {true_text}\nPred: {pred_text}", color=color, fontsize=10)
    
    plt.suptitle("Sample Predictions (Green: Correct, Red: Incorrect)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Show 10 random samples from the validation set
show_sample_results(model, X_val, y_val, num_images=10)