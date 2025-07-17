import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib
import sys

# Usage: python predict_multimodal.py <image_path> <zip> <beds> <baths> <sqft>

if len(sys.argv) != 6:
    print("Usage: python predict_multimodal.py <image_path> <zip> <beds> <baths> <sqft>")
    sys.exit(1)

image_path = sys.argv[1]
zip_code = int(sys.argv[2])
beds = int(sys.argv[3])
baths = int(sys.argv[4])
sqft = int(sys.argv[5])

# Load trained model
model = load_model("multimodal_house_price_model.h5")

# Load scaler and label encoder
scaler = joblib.load("scaler.save")
label_encoder = joblib.load("label_encoder.save")

# --- Prepare image ---
image = Image.open(image_path).convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
image_array = preprocess_input(image_array)
image_array = np.expand_dims(image_array, axis=0)

# --- Prepare structured data ---
data = pd.DataFrame([[zip_code, beds, baths, sqft]], columns=['zip', 'bed', 'bath', 'sqft'])
data_scaled = scaler.transform(data)

# --- Predict ---
predictions = model.predict([image_array, data_scaled])
probabilities = tf.nn.softmax(predictions[0]).numpy()

# Get top 2 classes
sorted_indices = np.argsort(probabilities)[::-1]
first_class = label_encoder.inverse_transform([sorted_indices[0]])[0]
second_class = label_encoder.inverse_transform([sorted_indices[1]])[0]
first_confidence = probabilities[sorted_indices[0]]
second_confidence = probabilities[sorted_indices[1]]

print(f"Most Likely Class: {first_class} with confidence {first_confidence:.4f}")
print(f"Second Likely Class: {second_class} with confidence {second_confidence:.4f}")
