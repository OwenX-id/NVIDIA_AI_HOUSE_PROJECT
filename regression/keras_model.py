from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

import sys
if len(sys.argv) < 2:
    print("Usage: python keras_model.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

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

print("Top Class:", class_name[2:].strip(), "| Confidence Score:", confidence_score)
print("Second Class:", second_class_name[2:].strip(), "| Confidence Score:", second_confidence_score)
print("Inflation-Adjusted Class:", adjusted_class_name[2:].strip())
