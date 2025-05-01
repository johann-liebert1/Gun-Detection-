import pandas as pd
import os

csv_path = 'Data/train.csv'
df = pd.read_csv(csv_path)
print(df.head())

# Create labels directory
os.makedirs('labels', exist_ok=True)

# Iterate through each image entry in the CSV
for index, row in df.iterrows():
    img_path = row['image_path']
    group_id = row['group_id']
    print(img_path)
    # Load image to get dimensions
    from PIL import Image
    img = Image.open(img_path)
    img_width, img_height = img.size
