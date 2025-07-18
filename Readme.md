Project: House Price Estimation using Keras Image Classification + Structured Data Adjustments
This project predicts house prices by combining:

An image classification model (keras_model.h5) to estimate a base price bracket.

Adjustments using structured data: city, square footage, bedrooms, and bathrooms.

Regional pricing data from Southern California (socal2.csv).

Features
✅ Uses Keras model to predict base price from an exterior house image.
✅ Adjusts price based on:

City median vs. state median price.

Square footage difference from median.

Bedroom and bathroom count differences.

Inflation multipliers based on prediction confidence class.

✅ Displays clear intermediate outputs for debugging and transparency.

Files
keras_model.py: Main script for prediction and adjustment.

keras_model.h5: Trained Keras classification model.

labels.txt: Labels corresponding to price brackets used by the model.

socal2.csv: Southern California housing data with city, price, sqft, bed, bath.
