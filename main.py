# Install Dependencies: Ensure you have rembg and Pillow installed.
# pip install rembg Pillow [in terminal]


import os

import PIL
from rembg import remove
from PIL import Image
import io
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Function to remove background and add a white background
def remove_background(image_path, output_path):
    with open(image_path, 'rb') as img_file:
        input_image = img_file.read()

    # Remove background using rembg
    output_image = remove(input_image)

    # Open the output image and paste it on a white background
    img_no_bg = Image.open(io.BytesIO(output_image)).convert("RGBA")
    white_bg = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))  # Create a white background
    white_bg.paste(img_no_bg, (0, 0), img_no_bg)  # Paste the image onto the white background

    # Convert to RGB (removes alpha channel) and save
    white_bg.convert('RGB').save(output_path)


# Function to process all images in the input folder
def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            remove_background(input_image_path, output_image_path)
            print(f'Processed: {filename}')


# Define input and output folder paths
input_folder = r'D:\guava\leaf_blight'  # Replace with your actual input folder path
output_folder = r'D:\bg_rm_guava\bg_rm_leaf_blight'  # Replace with your desired output folder path

# Run the process
process_images(input_folder, output_folder)
