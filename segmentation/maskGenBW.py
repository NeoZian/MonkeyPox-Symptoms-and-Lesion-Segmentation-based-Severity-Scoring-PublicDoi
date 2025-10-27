import os
import subprocess
import shutil
from PIL import Image
import numpy as np

# Define the paths
json_dir = 'data/json'  # Path to the folder with JSON files
output_dir = 'data/mask_png'  # Path to the folder to store PNG masks

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to convert mask to binary
def convert_to_binary_mask(mask_path):
    with Image.open(mask_path) as mask:
        # Convert to grayscale
        gray_mask = mask.convert('L')
        # Convert to numpy array
        gray_mask_np = np.array(gray_mask)
        # Create binary mask: 255 for object, 0 for background
        binary_mask_np = np.where(gray_mask_np > 0, 255, 0).astype(np.uint8)
        # Convert back to image
        binary_mask = Image.fromarray(binary_mask_np)
        # Save the binary mask
        binary_mask.save(mask_path)

# Iterate through all JSON files in the directory
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_dir, json_file)
        
        # Run the labelme_json_to_dataset command to convert JSON to a dataset folder
        output_dataset_path = os.path.join(output_dir, json_file.split('.')[0])
        subprocess.run(['labelme_json_to_dataset', json_path, '-o', output_dataset_path])
        
        # Copy and rename the mask image (label.png) to the output directory with the JSON file name
        mask_path = os.path.join(output_dataset_path, 'label.png')
        new_mask_name = json_file.replace('.json', '.png')
        new_mask_path = os.path.join(output_dir, new_mask_name)
        
        # Convert the mask to binary format
        convert_to_binary_mask(mask_path)
        
        # Move the mask image to the output directory with the desired name
        shutil.move(mask_path, new_mask_path)
        
        # Optionally remove the generated dataset folder after extracting the mask
        shutil.rmtree(output_dataset_path)

print("Conversion completed! All mask images are saved in:", output_dir)

