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
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Adjust for California 6-year housing inflation
inflation_multiplier = 1.504  # 50.4% increase
adjusted_index = min(int(index * inflation_multiplier), len(class_names) - 1)

# Print prediction and confidence score with inflation-adjusted class
print("Original Class:", class_name[2:], end="")
print(" Confidence Score:", confidence_score)

adjusted_class_name = class_names[adjusted_index]
print("Inflation-Adjusted Class:", adjusted_class_name[2:])

