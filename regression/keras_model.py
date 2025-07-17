from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import sys

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Input validation
if len(sys.argv) < 6:
    print("Usage: python keras_model.py <image_path> <city> <bedrooms> <bathrooms> <sqft>")
    sys.exit(1)

image_path = sys.argv[1]
city_input = sys.argv[2].strip()#.lower()
bedrooms_input = float(sys.argv[3])
bathrooms_input = float(sys.argv[4])
sqft_input = float(sys.argv[5])

# Process image
image = Image.open(image_path).convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# Predict with Keras
prediction = model.predict(data)
sorted_indices = np.argsort(prediction[0])[::-1]
index = sorted_indices[0]
second_index = sorted_indices[1]

class_name = class_names[index].strip()
confidence_score = prediction[0][index]
second_class_name = class_names[second_index].strip()
second_confidence_score = prediction[0][second_index]

inflation_multipliers = {
    0: 1.60,
    1: 1.55,
    2: 1.52,
    3: 1.48,
    4: 1.45,
    5: 1.40,
    6: 1.35
}
multiplier = inflation_multipliers.get(index, 1.50)
adjusted_price = (index + 1) * multiplier
adjusted_index = min(int(adjusted_price) - 1, len(class_names) - 1)
adjusted_class_name = class_names[adjusted_index].strip()

# Load socal data
socal_df = pd.read_csv("socal2.csv")
socal_df['citi'] = socal_df['citi'].astype(str).str.strip().str.lower()

median_price_state = socal_df['price'].median()
city_df = socal_df[socal_df['citi'] == city_input]

if not city_df.empty:
    median_price_city = city_df['price'].median()
    location_adjustment = median_price_city - median_price_state
    print(f"Location adjustment for {city_input.title()}: ${location_adjustment:,.0f}")
else:
    location_adjustment = 0
    print(f"City '{city_input}' not found in dataset. No location adjustment applied.")
# ADJUSTMENT CONTROL
SQFT_WEIGHT = 1.0
BEDROOM_WEIGHT = 1.0
BATHROOM_WEIGHT = 1.0

median_sqft_state = socal_df['sqft'].median()
median_bed_state = socal_df['bed'].median()
median_bath_state = socal_df['bath'].median()

# Debug view of medians for clarity
print(f"State median sqft: {median_sqft_state}, bed: {median_bed_state}, bath: {median_bath_state}")

# Sqft: 1 USD per sqft delta
sqft_adjustment = (sqft_input - median_sqft_state) * SQFT_WEIGHT

# Bedroom: 50,000 USD per bedroom delta
bedroom_adjustment = (bedrooms_input - median_bed_state) * 50000 * BEDROOM_WEIGHT

# Bathroom: 30,000 USD per bathroom delta
bathroom_adjustment = (bathrooms_input - median_bath_state) * 30000 * BATHROOM_WEIGHT

print(f"Sqft adjustment: ${sqft_adjustment:,.0f}")
print(f"Bedroom adjustment: ${bedroom_adjustment:,.0f}")
print(f"Bathroom adjustment: ${bathroom_adjustment:,.0f}")

# Base price from class
predicted_base_price = (index + 1) * 100000

# Apply all adjustments before inflation
adjusted_price_with_factors = (
    predicted_base_price +
    location_adjustment +
    sqft_adjustment +
    bedroom_adjustment +
    bathroom_adjustment
)

adjusted_price_with_factors *= multiplier

print(f"\nPredicted Base Price: ${predicted_base_price:,.0f}")
print(f"Adjusted Price (Location + Features + Inflation): ${adjusted_price_with_factors:,.0f}\n")