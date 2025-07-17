from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import sys

# ---------- CONFIG ----------

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

price_ranges = {
    0: 150_000,   # 100-200k midpoint
    1: 250_000,   # 200-300k
    2: 400_000,   # 300-500k
    3: 625_000,   # 500-750k
    4: 875_000,   # 750k-1m
    5: 1_125_000, # 1m-1.25m
    6: 1_400_000  # 1.25m+
}

inflation_multipliers = {
    0: 1.60,
    1: 1.55,
    2: 1.52,
    3: 1.48,
    4: 1.45,
    5: 1.40,
    6: 1.35
}

# ---------- CLI ARGS ----------
if len(sys.argv) < 3:
    print("Usage: python keras_model.py <image_path> <zip_code>")
    sys.exit(1)

image_path = sys.argv[1]
zip_code_input = sys.argv[2]

# ---------- IMAGE PROCESSING ----------
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open(image_path).convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# ---------- CSV COUNTY & PRICE PROCESSING ----------
socal_df = pd.read_csv("socal2.csv")
socal_df.columns = socal_df.columns.str.lower().str.strip()

# Infer likely column names
zip_col = [col for col in socal_df.columns if 'zip' in col][0]
county_col = [col for col in socal_df.columns if 'county' in col][0]
price_col = [col for col in socal_df.columns if 'price' in col][0]

# Ensure types
socal_df[zip_col] = socal_df[zip_col].astype(str).str[:5]

# Median price for California
median_ca_price = socal_df[price_col].median()

# Lookup county for input zip
row = socal_df[socal_df[zip_col] == zip_code_input]
if row.empty:
    print(f"Zip code {zip_code_input} not found in dataset.")
    sys.exit(1)
county_name = row[county_col].iloc[0]

# Median price for the county
median_county_price = socal_df[socal_df[county_col] == county_name][price_col].median()

# Compute price difference
price_delta = median_county_price - median_ca_price

# ---------- KERAS PREDICTION ----------
prediction = model.predict(data)
sorted_indices = np.argsort(prediction[0])[::-1]
index = sorted_indices[0]
second_index = sorted_indices[1]

top_class_name = class_names[index].strip()[2:]
confidence_score = prediction[0][index]

second_class_name = class_names[second_index].strip()[2:]
second_confidence_score = prediction[0][second_index]

# ---------- ADJUSTED PRICE ----------
predicted_price = price_ranges.get(index, 400_000)
adjusted_price = predicted_price + price_delta

# Inflation adjustment
inflation_multiplier = inflation_multipliers.get(index, 1.5)
inflation_adjusted_price = adjusted_price * inflation_multiplier

# ---------- OUTPUT ----------
print(f"Top Class: {top_class_name} | Confidence: {confidence_score:.4f}")
print(f"Second Class: {second_class_name} | Confidence: {second_confidence_score:.4f}")
print(f"Predicted Price Midpoint: ${predicted_price:,.0f}")
print(f"Median County Price: ${median_county_price:,.0f}")
print(f"Median California Price: ${median_ca_price:,.0f}")
print(f"Adjusted Price (with county delta): ${adjusted_price:,.0f}")
print(f"Inflation Adjusted Price: ${inflation_adjusted_price:,.0f}")
