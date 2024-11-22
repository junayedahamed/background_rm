import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Function to load images and split the dataset into training, validation, and testing
def prepare_data(parent_folder_path):
    # Create an ImageDataGenerator with rescaling and split for training/validation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 80% for training, 20% for validation
    )

    # Training set (80%)
    train_generator = datagen.flow_from_directory(
        parent_folder_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # 80% for training
    )

    # Validation set (20% of training data)
    validation_generator = datagen.flow_from_directory(
        parent_folder_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # 20% for validation
    )

    # For testing, we need to load the data manually or create a separate test set.
    # One way to do it is to split your directory structure manually, or use a different method for test data
    test_datagen = ImageDataGenerator(rescale=1./255)  # Test data generator, no splitting
    test_generator = test_datagen.flow_from_directory(
        parent_folder_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Don't shuffle test data, keep it consistent
    )

    return train_generator, validation_generator, test_generator

# Build a model using Vision Transformer (ViT) from TensorFlow Hub
def build_model(num_classes):
    # Load the pre-trained Vision Transformer (ViT) model from TensorFlow Hub
    vit_model_url = "https://tfhub.dev/google/vit_b32_patch16_224/1"  # A different version of ViT that should work
    base_model = hub.KerasLayer(vit_model_url, input_shape=(150, 150, 3))

    # Build the custom model using the ViT base model
    model = Sequential([
        base_model,  # Add the ViT model
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation for classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot training and validation accuracy
def plot_accuracy(history):
    # Plot accuracy and loss graphs
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Main execution
def main(parent_folder_path):
    # Prepare data
    train_generator, validation_generator, test_generator = prepare_data(parent_folder_path)

    # Get number of classes from the generator
    num_classes = len(train_generator.class_indices)

    # Build and compile model
    model = build_model(num_classes)

    # Train model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Plot accuracy and loss charts
    plot_accuracy(history)

# Example of calling the function with the parent folder path
if __name__ == "__main__":
    parent_folder_path = "D:\\Vegetable identification dataset\\Original dataset"  # Replace with your actual path
    main(parent_folder_path)
