from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import pandas as pd
from uszipcode import SearchEngine

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

import sys
if len(sys.argv) < 3:
    print("Usage: python keras_model.py <image_path> <zip_code>")
    sys.exit(1)

image_path = sys.argv[1]
zip_code_input = sys.argv[2]

# Open the provided image
image = Image.open(image_path).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
# Get indices of sorted predictions (highest first)
sorted_indices = np.argsort(prediction[0])[::-1]
index = sorted_indices[0]
second_index = sorted_indices[1]

# Retrieve class names and confidence scores
class_name = class_names[index]
confidence_score = prediction[0][index]

second_class_name = class_names[second_index]
second_confidence_score = prediction[0][second_index]

# Mapping from index to inflation multiplier
inflation_multipliers = {
    0: 1.60,  # 100-200k
    1: 1.55,  # 200-300k
    2: 1.52,  # 300-500k
    3: 1.48,  # 500-750k
    4: 1.45,  # 750k-1m
    5: 1.40,  # 1m-1.25m
    6: 1.35   # 1.25m+
}

# Retrieve appropriate multiplier for predicted class
multiplier = inflation_multipliers.get(index, 1.50)  # default if out of mapping

# Estimate adjusted price class
adjusted_price = (index + 1) * multiplier  # simple inflation estimate
adjusted_index = min(int(adjusted_price) - 1, len(class_names) - 1)

adjusted_class_name = class_names[adjusted_index]

# ========== New block: Median price by zip code county ==========

try:
    # Load your CSV file with zip codes and prices
    df = pd.read_csv("socal2.csv")

    # Ensure columns are correctly named
    df.columns = df.columns.str.lower().str.strip()
    if 'zip_code' not in df.columns or 'price' not in df.columns:
        print("CSV missing required 'zip_code' and 'price' columns.")
    else:
        # Initialize zip code search engine
        search = SearchEngine(simple_zipcode=True)

        # Map zip codes to counties
        def get_county(zip_code):
            zipcode_info = search.by_zipcode(str(int(zip_code)))
            return zipcode_info.county if zipcode_info else None

        # Add county column
        df['county'] = df['zip_code'].apply(get_county)

        # Drop rows without county
        df = df.dropna(subset=['county'])

        # Group by county and compute median price
        median_prices = df.groupby('county')['price'].median().to_dict()

        # Get county for the provided test zip code
        test_zip_info = search.by_zipcode(str(int(zip_code_input)))
        test_county = test_zip_info.county if test_zip_info else None

        if test_county and test_county in median_prices:
            print(f"Median price in {test_county} county: ${median_prices[test_county]:,.0f}")
        else:
            print(f"Could not find median price for county of zip code {zip_code_input}.")

except Exception as e:
    print(f"Error processing county median price: {e}")

# ========== End new block ==========

# Print final predictions
print("Top Class:", class_name[2:].strip(), "| Confidence Score:", confidence_score)
print("Second Class:", second_class_name[2:].strip(), "| Confidence Score:", second_confidence_score)
print("Inflation-Adjusted Class:", adjusted_class_name[2:].strip())
