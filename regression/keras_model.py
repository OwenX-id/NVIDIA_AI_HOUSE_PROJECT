from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import sys

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Input handling
if len(sys.argv) < 3:
    print("Usage: python keras_model.py <image_path> <city>")
    sys.exit(1)

image_path = sys.argv[1]
city_input = sys.argv[2].strip().lower()

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

# Inflation multipliers
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

# Ensure consistent case for city matching
socal_df['citi'] = socal_df['citi'].astype(str).str.strip().str.lower()

# Median price calculations
median_price_state = socal_df['price'].median()

# Filter for the input city
city_df = socal_df[socal_df['citi'] == city_input]

if not city_df.empty:
    median_price_city = city_df['price'].median()
    location_adjustment = median_price_city - median_price_state
    print(f"Location adjustment for {city_input.title()}: ${location_adjustment:,.0f}")
else:
    location_adjustment = 0
    print(f"City '{city_input}' not found in dataset. No location adjustment applied.")

# ------------------------------------
# ADJUSTMENT CONTROL: Adjustment weights
SQFT_WEIGHT = 0.5
BEDROOM_WEIGHT = 0.25
BATHROOM_WEIGHT = 0.25
# ------------------------------------

# Compute state and city medians for sqft, bedrooms, bathrooms
median_sqft_state = socal_df['sqft'].median()
median_bed_state = socal_df['bed'].median()
median_bath_state = socal_df['bath'].median()

if not city_df.empty:
    median_sqft_city = city_df['sqft'].median()
    median_bed_city = city_df['bed'].median()
    median_bath_city = city_df['bath'].median()

    # Adjustments
    sqft_adjustment = (median_sqft_city - median_sqft_state) * SQFT_WEIGHT
    bedroom_adjustment = (median_bed_city - median_bed_state) * 50000 * BEDROOM_WEIGHT  # each bed ~50k value baseline
    bathroom_adjustment = (median_bath_city - median_bath_state) * 30000 * BATHROOM_WEIGHT  # each bath ~30k value baseline

    print(f"Sqft adjustment: ${sqft_adjustment:,.0f}")
    print(f"Bedroom adjustment: ${bedroom_adjustment:,.0f}")
    print(f"Bathroom adjustment: ${bathroom_adjustment:,.0f}")

else:
    sqft_adjustment = 0
    bedroom_adjustment = 0
    bathroom_adjustment = 0

# Example: simulate predicted base price from class
predicted_base_price = (index + 1) * 100000  # simplistic mapping for clarity

# Apply all adjustments before inflation
adjusted_price_with_factors = (
    predicted_base_price +
    location_adjustment +
    sqft_adjustment +
    bedroom_adjustment +
    bathroom_adjustment
)

# Apply inflation adjustment
adjusted_price_with_factors *= multiplier

print(f"\nPredicted Base Price: ${predicted_base_price:,.0f}")
print(f"Adjusted Price (Location + Features + Inflation): ${adjusted_price_with_factors:,.0f}\n")

print("Top Class:", class_name, "| Confidence Score:", confidence_score)
print("Second Class:", second_class_name, "| Confidence Score:", second_confidence_score)
print("Inflation-Adjusted Class:", adjusted_class_name)
