import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm
import cv2

# Paths for input and output directories
input_dir = Path("D:\\Guava Dataset\\Guava DataSet\\bg removed healthy leaves")
output_dir = Path("D:\\guava final ds\\healthy")
output_dir.mkdir(parents=True, exist_ok=True)

# Total number of augmented images needed
target_count = 2001
current_count = len(list(input_dir.glob("*.jpg")))  # Assumes .jpg format; adjust as needed
augmentation_needed = target_count - current_count


# Function definitions for augmentations
def flip_image(image):
    return cv2.flip(image, 1)  # Horizontal flip


def rotate_image(image):
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))


def scale_image(image):
    scale_factor = random.uniform(0.8, 1.2)
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)


def translate_image(image):
    tx = random.randint(-20, 20)
    ty = random.randint(-20, 20)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def crop_image(image):
    h, w = image.shape[:2]
    start_x = random.randint(0, w // 10)
    start_y = random.randint(0, h // 10)
    end_x = w - random.randint(0, w // 10)
    end_y = h - random.randint(0, h // 10)
    return image[start_y:end_y, start_x:end_x]


def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    return np.array(enhancer.enhance(random.uniform(0.7, 1.3)))


def adjust_contrast(image):
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    return np.array(enhancer.enhance(random.uniform(0.7, 1.3)))


# List of augmentations to apply randomly
augmentations = [
    flip_image, rotate_image, scale_image, translate_image, crop_image,
    add_noise, adjust_brightness, adjust_contrast
]

# Apply augmentations and save images
image_files = list(input_dir.glob("*.jpg"))
for i in tqdm(range(augmentation_needed)):
    img_path = random.choice(image_files)
    img = cv2.imread(str(img_path))

    # Apply a random augmentation
    augmentation = random.choice(augmentations)
    aug_img = augmentation(img)

    # Resize to 800x800 to maintain quality and reduce size
    target_size = 800
    aug_img = cv2.resize(aug_img, (target_size, target_size))

    # Convert image to PIL for DPI setting
    pil_img = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))

    # Save the image while keeping the file size under 100KB
    aug_img_path = output_dir / f"aug_{i}.jpg"
    quality = 95  # Start with high quality

    # Compress image iteratively while checking file size
    while True:
        pil_img.save(aug_img_path, "JPEG", quality=quality, dpi=(120, 120))
        if os.path.getsize(aug_img_path) <= 100 * 1024 or quality <= 60:
            break
        quality -= 5