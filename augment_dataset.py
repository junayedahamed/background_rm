import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
from tqdm import tqdm

def horizontal_flip(image):
    """Horizontal flip augmentation"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image):
    """Vertical flip augmentation"""
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_image(image, max_angle=15):
    """Rotate image by random angle (maintains 224x224)"""
    angle = random.uniform(-max_angle, max_angle)
    # Rotate and maintain size
    rotated = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    return rotated

def adjust_brightness(image, factor_range=(0.8, 1.2)):
    """Adjust brightness while maintaining dimensions"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor_range=(0.8, 1.2)):
    """Adjust contrast while maintaining dimensions"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor_range=(0.8, 1.2)):
    """Adjust saturation while maintaining dimensions"""
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def add_noise(image, noise_factor=0.05):
    """Add Gaussian noise while maintaining dimensions"""
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def random_crop_and_resize(image, crop_range=(0.85, 1.0)):
    """Random crop and resize back to 224x224"""
    crop_factor = random.uniform(crop_range[0], crop_range[1])
    w, h = image.size
    new_w = int(w * crop_factor)
    new_h = int(h * crop_factor)
    
    # Random crop position
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    
    # Crop and resize back to original size
    cropped = image.crop((left, top, left + new_w, top + new_h))
    resized = cropped.resize((224, 224), Image.Resampling.LANCZOS)
    return resized

def augment_image(image, augmentation_type=None):
    """
    Apply a single augmentation to an image.
    Maintains 224x224 dimensions.
    """
    if augmentation_type is None:
        # Random augmentation
        augmentation_type = random.choice([
            'horizontal_flip', 'vertical_flip', 'rotate', 
            'brightness', 'contrast', 'saturation', 
            'noise', 'crop_resize'
        ])
    
    if augmentation_type == 'horizontal_flip':
        return horizontal_flip(image)
    elif augmentation_type == 'vertical_flip':
        return vertical_flip(image)
    elif augmentation_type == 'rotate':
        return rotate_image(image)
    elif augmentation_type == 'brightness':
        return adjust_brightness(image)
    elif augmentation_type == 'contrast':
        return adjust_contrast(image)
    elif augmentation_type == 'saturation':
        return adjust_saturation(image)
    elif augmentation_type == 'noise':
        return add_noise(image)
    elif augmentation_type == 'crop_resize':
        return random_crop_and_resize(image)
    else:
        return image

def augment_single_image(image_path, output_path, num_augmentations=1):
    """
    Augment a single image multiple times.
    
    Args:
        image_path: Path to input image
        output_path: Base path for output (without extension)
        num_augmentations: Number of augmented versions to create
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Ensure image is 224x224
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Create augmented versions
        for i in range(num_augmentations):
            aug_img = augment_image(img.copy())
            
            # Ensure output is still 224x224
            if aug_img.size != (224, 224):
                aug_img = aug_img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Save augmented image
            aug_output_path = f"{output_path}_aug_{i+1}.jpg"
            aug_img.save(aug_output_path, "JPEG", quality=95)
        
        return True
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False

def batch_augment(
    input_folder,
    output_folder,
    num_augmentations_per_image=3,
    target_count=None
):
    """
    Batch augment all images in a folder.
    
    Args:
        input_folder: Path to folder containing input images (224x224)
        output_folder: Path to folder where augmented images will be saved
        num_augmentations_per_image: Number of augmented versions per image (used when target_count is None)
        target_count: Total number of images needed (original + augmented). If None, augments all images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files from input folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    input_image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not input_image_files:
        print(f"No image files found in {input_folder}")
        return
    
    # Current count = only input folder images
    current_count = len(input_image_files)
    
    print(f"Current count (input images): {current_count}")
    print(f"Target size: 224x224 (maintained)")
    
    # Mode 1: Target count mode (like old augmentation.py)
    if target_count is not None:
        # Calculate how many augmented images need to be created
        # Formula: target_count - current_count = augmented images to create
        augmentation_needed = max(0, target_count - current_count)
        
        print(f"Target count: {target_count}")
        print(f"Current count (input images): {current_count}")
        print(f"Augmented images to create: {augmentation_needed}")
        print("-" * 50)
        
        if augmentation_needed <= 0:
            print("Target count already reached! No augmentation needed.")
            return
        
        successful = 0
        failed = 0
        aug_counter = 0
        
        # Randomly select and augment images until target is reached
        for i in tqdm(range(augmentation_needed), desc="Creating augmented images"):
            # Randomly select an image
            img_filename = random.choice(input_image_files)
            input_path = os.path.join(input_folder, img_filename)
            
            # Get base name without extension
            name_without_ext = os.path.splitext(img_filename)[0]
            
            # Create unique augmented filename
            aug_counter += 1
            output_filename = f"{name_without_ext}_aug_{aug_counter}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                # Load and ensure 224x224
                img = Image.open(input_path).convert('RGB')
                if img.size != (224, 224):
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Apply random augmentation
                aug_img = augment_image(img.copy())
                
                # Ensure still 224x224
                if aug_img.size != (224, 224):
                    aug_img = aug_img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Save
                aug_img.save(output_path, "JPEG", quality=95)
                successful += 1
            except Exception as e:
                print(f"Error augmenting {img_filename}: {e}")
                failed += 1
        
        print("-" * 50)
        print(f"Augmentation complete!")
        print(f"Augmented images created: {successful}")
        print(f"Failed: {failed}")
        print(f"Original images: {current_count}")
        print(f"Total images after augmentation: {current_count + successful}")
    
    # Mode 2: Augment all images N times each
    else:
        print(f"Augmentations per image: {num_augmentations_per_image}")
        print("-" * 50)
        
        successful = 0
        failed = 0
        total_augmented = 0
        
        for filename in tqdm(input_image_files, desc="Augmenting images"):
            input_path = os.path.join(input_folder, filename)
            
            # Get base name without extension
            name_without_ext = os.path.splitext(filename)[0]
            output_base_path = os.path.join(output_folder, name_without_ext)
            
            # Augment the image
            if augment_single_image(input_path, output_base_path, num_augmentations_per_image):
                successful += 1
                total_augmented += num_augmentations_per_image
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Augmentation complete!")
        print(f"Images processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Total augmented images created: {total_augmented}")
        print(f"Original images: {len(input_image_files)}")
        print(f"Total images after augmentation: {len(input_image_files) + total_augmented}")


if __name__ == "__main__":
    # ============================================
    # CONFIGURATION - Adjust these values
    # ============================================
    
    # Input and output folders
    input_folder = r'D:\guava final ds\nutritional disorder'  # Preprocessed images folder (224x224)
    output_folder = r'D:\augmented\nutritional disorder'  # Output folder (can be same or different)
    
    # ============================================
    # AUGMENTATION MODE SELECTION
    # ============================================
    # Choose ONE of the following modes:
    
    # MODE 1: Target Count (like old augmentation.py)
    # Specify TOTAL dataset size you want (original + augmented)
    # Script will calculate: target_count - current_count = augmented images to create
    target_count = 2000  # Total dataset size (original + augmented). Set to None to use MODE 2 instead
    
    # MODE 2: Augment All Images N Times
    # Number of augmented versions per image (used when target_count is None)
    num_augmentations_per_image = 3  # Each image will have 3 augmented versions
    
    # ============================================
    # Start augmentation
    # ============================================
    batch_augment(
        input_folder=input_folder,
        output_folder=output_folder,
        num_augmentations_per_image=num_augmentations_per_image,
        target_count=target_count  # Set to None to augment all images N times each
    )

