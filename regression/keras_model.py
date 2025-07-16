from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python keras_model.py test_image.png")
    sys.exit(1)

image_path = sys.argv[1]

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# Predict
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Output
print(f"Class: {class_name.strip()}")
print(f"Confidence Score: {confidence_score:.4f}")
