import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Function to split the dataset into train (70%), validation (15%), and test (15%)
def split_dataset(original_dataset_path, base_output_path):
    """
    Splits the dataset into training, validation, and testing datasets.
    """
    train_path = os.path.join(base_output_path, 'train')
    validation_path = os.path.join(base_output_path, 'validation')
    test_path = os.path.join(base_output_path, 'test')

    # Create output directories
    for path in [train_path, validation_path, test_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Split dataset
    for class_folder in os.listdir(original_dataset_path):
        class_path = os.path.join(original_dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Get all file paths in the class directory
        file_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Split into training (70%) and temp (30%)
        train_files, temp_files = train_test_split(file_paths, test_size=0.3, random_state=42)

        # Further split temp into validation (15%) and testing (15%)
        validation_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Copy files to their respective directories
        for file_list, target_dir in zip([train_files, validation_files, test_files], [train_path, validation_path, test_path]):
            class_output_dir = os.path.join(target_dir, class_folder)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)
            for file in file_list:
                shutil.copy(file, class_output_dir)

    return train_path, validation_path, test_path

# Function to load images from the respective folders for train, validation, and test sets
def prepare_data(train_dir, validation_dir, test_dir):
    # Image data generators for each dataset
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Create data generators for train, validation, and test datasets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator

# Build a CNN model for image classification
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(6, activation='softmax')  # Adjust for the number of classes in your dataset
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot accuracy and loss charts
def plot_metrics(history):
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# Main execution
def main(original_dataset_path, base_output_path):
    # Split the dataset into train, validation, and test sets
    train_dir, validation_dir, test_dir = split_dataset(original_dataset_path, base_output_path)

    # Prepare data
    train_generator, validation_generator, test_generator = prepare_data(train_dir, validation_dir, test_dir)

    # Build model
    model = build_model()

    # Train model
    history = model.fit(
        train_generator,
        epochs=20,  # Increased to 20 epochs
        validation_data=validation_generator
    )

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Plot training and validation metrics
    plot_metrics(history)

# Example of calling the function with the dataset paths
if __name__ == "__main__":
    # Provide the original dataset path and base output directory for splitting
    original_dataset_path = "D:\\Vegetable identification dataset\\Original dataset"  # Replace with your actual path
    base_output_path = "D:\\base output"     # Path to store split datasets
    main(original_dataset_path, base_output_path)
