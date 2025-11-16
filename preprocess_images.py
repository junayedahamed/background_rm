import os
import io
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm

# Try to import rembg for background removal (optional)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Note: rembg not installed. Background removal will not be available.")
    print("Install it with: pip install rembg")
    # Create a dummy remove function to avoid errors
    remove = None

# Try to import pillow-heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files may not be supported.")
    print("Install it with: pip install pillow-heif")

# Prevent DecompressionBombError for very large images
Image.MAX_IMAGE_PIXELS = None

def preprocess_image(
    image_path, 
    output_path, 
    target_size=(224, 224),
    brightness_factor=1.0,
    remove_bg=False,
    maintain_aspect_ratio=False
):
    """
    Preprocess a single image: resize, adjust brightness, and optionally remove background.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        target_size: Target dimensions (width, height). Default: (224, 224)
        brightness_factor: Brightness adjustment factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
                           Recommended range: 0.8 to 1.2
        remove_bg: If True, remove background and apply white background
        maintain_aspect_ratio: If True, maintain aspect ratio and pad with white background
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Remove background if requested
        if remove_bg:
            if not REMBG_AVAILABLE:
                raise ImportError(
                    "rembg is required for background removal. "
                    "Install it with: pip install rembg"
                )
            
            with open(image_path, 'rb') as img_file:
                input_data = img_file.read()
            
            # Remove background
            output_data = remove(input_data)
            
            # Convert output bytes to image
            img = Image.open(io.BytesIO(output_data)).convert("RGBA")
            
            # Create white background
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            white_bg.paste(img, (0, 0), img)
            img = white_bg.convert("RGB")
        else:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Adjust brightness
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        # Resize image
        if maintain_aspect_ratio:
            # Maintain aspect ratio and pad with white background
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with white background
            new_img = Image.new("RGB", target_size, (255, 255, 255))
            
            # Calculate position to center the image
            x_offset = (target_size[0] - img.size[0]) // 2
            y_offset = (target_size[1] - img.size[1]) // 2
            
            # Paste the resized image onto the white background
            new_img.paste(img, (x_offset, y_offset))
            img = new_img
        else:
            # Resize without maintaining aspect ratio (may distort image)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Ensure image is in RGB mode before saving (required for JPEG)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Ensure output path has .jpg extension
        if not output_path.lower().endswith('.jpg'):
            output_path = os.path.splitext(output_path)[0] + '.jpg'
        
        # Save processed image as JPEG format (.jpg)
        img.save(output_path, "JPEG", quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def batch_preprocess(
    input_folder,
    output_folder,
    target_size=(224, 224),
    brightness_factor=1.0,
    remove_bg=False,
    maintain_aspect_ratio=False
):
    """
    Batch process all images in a folder.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where processed images will be saved
        target_size: Target dimensions (width, height). Default: (224, 224)
        brightness_factor: Brightness adjustment factor (1.0 = no change)
        remove_bg: If True, remove background and apply white background
        maintain_aspect_ratio: If True, maintain aspect ratio and pad with white background
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files (including HEIC)
    image_extensions = ('.png', '.jpg', '.jpeg', '.heic', '.PNG', '.JPG', '.JPEG', '.HEIC')
    image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))
    ]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Target size: {target_size}")
    print(f"Brightness factor: {brightness_factor}")
    print(f"Remove background: {remove_bg}")
    print(f"Maintain aspect ratio: {maintain_aspect_ratio}")
    print("-" * 50)
    
    # Process each image with progress bar
    successful = 0
    failed = 0
    
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        
        # Keep original filename, ensure output is always .jpg format
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}.jpg"  # Always .jpg extension
        output_path = os.path.join(output_folder, output_filename)
        
        if preprocess_image(
            input_path,
            output_path,
            target_size=target_size,
            brightness_factor=brightness_factor,
            remove_bg=remove_bg,
            maintain_aspect_ratio=maintain_aspect_ratio
        ):
            successful += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    # ============================================
    # CONFIGURATION - Adjust these values
    # ============================================
    
    # Input and output folders
    input_folder = r'D:\Guava Dataset\Guava DataSet\nutritional disorder'  # Replace with your input folder path
    output_folder = r'D:\guava final ds\nutritional disorder'  # Replace with your output folder path
    
    # Image dimensions (width, height)
    target_size = (224, 224)  # Standard size for most CNN models
    
    # Brightness adjustment
    # 1.0 = no change, 1.2 = 20% brighter, 0.8 = 20% darker
    # Set to 1.05 for research paper - provides subtle enhancement for better feature extraction
    brightness_factor = 1.05  # Research-appropriate value (5% brightness enhancement)
    
    # Background removal
    remove_bg = False  # Set to True if you want to remove background
    
    # Aspect ratio
    maintain_aspect_ratio = True  # Set to True to maintain aspect ratio (recommended)
    
    # ============================================
    # Start processing
    # ============================================
    batch_preprocess(
        input_folder=input_folder,
        output_folder=output_folder,
        target_size=target_size,
        brightness_factor=brightness_factor,
        remove_bg=remove_bg,
        maintain_aspect_ratio=maintain_aspect_ratio
    )

