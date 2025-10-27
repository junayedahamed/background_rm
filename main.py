import os
import io
from rembg import remove
from PIL import Image

# Prevent DecompressionBombError for very large images
Image.MAX_IMAGE_PIXELS = None

# Function to remove background and apply white background
def remove_background(image_path, output_path):
    try:
        with open(image_path, 'rb') as img_file:
            input_data = img_file.read()

        # Remove background
        output_data = remove(input_data)

        # Convert output bytes to image
        img_no_bg = Image.open(io.BytesIO(output_data)).convert("RGBA")

        # Create white background
        white_bg = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))
        white_bg.paste(img_no_bg, (0, 0), img_no_bg)

        # Convert to RGB and save
        white_bg.convert("RGB").save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to batch process images in folder
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            print(f'Processing: {filename}')
            remove_background(input_image_path, output_image_path)

# 🟩 Replace these paths with your actual folders
input_folder = r'D:\guava\red_need_check'
output_folder = r'D:\bg_rm_guava\bg_rm_leaf_blight'

# Start processing
process_images(input_folder, output_folder)
