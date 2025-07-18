

## README

# My Jetson-Based Project 🚀

![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-green?logo=nvidia)
![NVIDIA](https://img.shields.io/badge/Powered_by-NVIDIA-green?logo=nvidia)

This project runs on an NVIDIA Jetson Xavier using TensorRT for fast inference...


# 🏠 House Price Estimation with Image + Data Adjustments

This project predicts **house prices** by combining:

* An **image of the house** using a **Keras classification model**.
* **City, square footage, bedrooms, and bathrooms** for detailed adjustments.
* Real Southern California housing data for location-specific tuning.

It outputs a **realistic adjusted price estimate** based on the image and input features.

---

## 📂 Project Files

* `keras_model.py` – Main script to run predictions.
* `keras_model.h5` – Pre-trained Keras image classification model.
* `labels.txt` – Labels mapping model output to base price classes.
* `socal2.csv` – CSV file with Southern California housing data (city, price, sqft, bedrooms, bathrooms).

---

## ⚙️ Requirements

* Python 3.x
* Install dependencies:

```bash
pip install tensorflow keras numpy pandas Pillow
```

---

## 🚀 How to Run

Use the following command:

```bash
python keras_model.py <image_path> <city> <bedrooms> <bathrooms> <sqft>
```

### Arguments:

* `<image_path>`: Path to the **house exterior image** (e.g., `house.jpg`).
* `<city>`: City name (e.g., `"Los Angeles"`).
* `<bedrooms>`: Number of bedrooms (e.g., `3`).
* `<bathrooms>`: Number of bathrooms (e.g., `2`).
* `<sqft>`: Square footage of the house (e.g., `1600`).

### Example:

```bash
python keras_model.py house.jpg "Los Angeles" 3 2 1600
```

---

## 🖼️ What the Script Does

✅ **Loads the pre-trained Keras model** and `labels.txt`.
✅ **Loads Southern California data** to adjust for city-specific prices.
✅ **Processes the image**:

* Resizes to `224x224`.
* Normalizes for the model.

✅ **Predicts a base price bracket** using the image classifier.
✅ **Adjusts the price** based on:

* City median vs. state median price.
* Square footage difference.
* Bedroom and bathroom differences.
* Inflation multipliers.

✅ **Prints a clear, step-by-step output** so you understand each adjustment.

---

## 🖨️ Sample Output

When you run the script, you will see:

```
Median price in Los Angeles county: $725,000
Median price in California: $685,000
Location Adjustment: $12,000

State median sqft: 1500.0, bed: 3.0, bath: 2.0
Sqft adjustment: $5,000
Bedroom adjustment: $0
Bathroom adjustment: $0

Adjusted Price (Location + Features + Inflation): $412,000
```

This shows:

* Location adjustment based on your city.
* Sqft, bedroom, and bathroom adjustments from the state median.
* Final **adjusted estimated price** for your house.

---

## 🛠️ Customization & Tuning

* Adjust `SQFT_WEIGHT`, `BEDROOM_WEIGHT`, and `BATHROOM_WEIGHT` in `keras_model.py` to fit your region or preferences.
* Update the `inflation_multipliers` dictionary to reflect local inflation or market volatility.
* Replace `keras_model.h5` and `labels.txt` with your own trained models for different regions or property types.

---

## ⚡ Notes

✅ Ensure **all files are in the same directory** before running:
`keras_model.py`, `keras_model.h5`, `labels.txt`, `socal2.csv`.

✅ The image should be a **clear exterior image** of the house for best results.

✅ The city name should match a city in `socal2.csv` for location adjustment. If not found, the script will continue without location-based adjustments.

✅ All outputs are printed to the terminal for **transparency and learning**.

---

## ✨ Why This Project?

This project demonstrates **integrating deep learning (image classification) with structured data** to estimate house prices more realistically, making it a powerful tool for:

* Real estate exploration.
* Market analysis projects.
* Portfolio demonstrations for **applied ML + data science.**


