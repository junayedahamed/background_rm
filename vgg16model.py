
# dataset_path = 'D:\\Vegetable identification datasetx\\Original dataset'  # Update to your actual path

from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report



# Step 2: Set up paths and parameters
dataset_path = 'D:\\Vegetable identification datasetx\\Original dataset' # path to the folder containing 'Eggplant' and 'Not Eggplant' subfolders.
img_size = (224, 224)
batch_size = 32
epochs = 30
learning_rate = 1e-5

# Step 3: Data loading without augmentation
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary', # or 'categorical' if you have more than two classes
    subset='training'
)

val_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary', # or 'categorical' if you have more than two classes
    subset='validation'
)



# # Step 2: Set up paths and parameters
# dataset_path = '/content/drive/MyDrive/vegetable data/Original dataset/Eggplant' # path diba folder er .
# img_size = (224, 224)
# batch_size = 32
# epochs = 30
# learning_rate = 1e-5

# # Step 3: Data loading without augmentation
# data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# train_data = data_gen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training'
# )

# val_data = data_gen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )

# Step 4: Build and fine-tune the VGG16 model
def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Unfreeze some of the last few layers for fine-tuning
    for layer in base_model.layers[:-4]:  # Unfreeze last 4 layers
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.7)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_vgg16()

# Step 5: Train the model with early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Step 6: Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_history(history)

# Step 7: Model evaluation
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Step 8: Prediction and metrics calculation
val_data.reset()
true_labels = val_data.classes
predictions = (model.predict(val_data) > 0.5).astype("int32").flatten()

# Confusion matrix and classification metrics
conf_matrix = confusion_matrix(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(true_labels, predictions, target_names=['Vegetable', 'Not Vegetable']))