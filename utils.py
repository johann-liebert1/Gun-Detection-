import os
import cv2
import pandas as pd
import numpy as np

# Path to the CSV file
csv_file = 'Data/train.csv'

# Load the CSV file
df = pd.read_csv(csv_file)

# Folder to save processed images (if needed)
processed_image_folder = 'Processed'
os.makedirs(processed_image_folder, exist_ok=True)

# Iterate over each row in the CSV
for index, row in df.iterrows():
    image_path = row['image_path']  # Replace 'image_path' with the actual column name

    # Check if the image exists at the specified path
    if os.path.exists(image_path):
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Resize the image to 32x32
        resized_image = cv2.resize(image, (32, 32))

        # Normalize the image (optional: normalize pixel values to [0, 1])
        resized_image = resized_image / 255.0

        # Optionally, save the resized image (if you want to keep them on disk)
        save_path = os.path.join(processed_image_folder, f'processed_{index}.jpg')
        cv2.imwrite(save_path, (resized_image * 255).astype(np.uint8))  # Save as original pixel range (0-255)

        # Store the new image path in the DataFrame
        df.loc[index, 'processed_image_path'] = save_path

    else:
        print(f"Image not found at {image_path}")

# Save the updated DataFrame with the processed image paths
df.to_csv('Data/updated.csv', index=False)
