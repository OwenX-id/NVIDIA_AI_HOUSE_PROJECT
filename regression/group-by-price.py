import os
import pandas as pd
import shutil

# Define paths
csv_path = 'dataset/socal2.csv'
images_dir = 'dataset/socal2/socal_pics'   # Adjust if needed
output_dir = 'teachable_machine_batches'

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Price bins
bins = [100000, 200000, 300000, 500000, 750000, 1000000, 1250000, float('inf')]
labels = [
    '100-200k', '200-300k', '300-500k', '500-750k',
    '750k-1m', '1m-1.25m', '1.25m+'
]

# Assign bin labels
df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)

# Group images into bins
for idx, row in df.iterrows():
    img_filename = f"{idx}.jpg"
    img_path = os.path.join(images_dir, img_filename)

    price_label = row['price_bin']

    if pd.isna(price_label):
        print(f"[Warning] Price for index {idx} does not fit any bin. Skipping.")
        continue

    bin_folder = os.path.join(output_dir, str(price_label))
    os.makedirs(bin_folder, exist_ok=True)

    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(bin_folder, img_filename))
    else:
        print(f"[Warning] Image {img_path} not found. Skipping.")

print("✅ Images grouped into folders by price range for Teachable Machine.")
