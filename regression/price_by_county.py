import tabula
import pandas as pd

# --- Step 1: Extract ZIP-to-County table from your PDF ---
pdf_path = "california-zip-codes.pdf"  # Replace with your PDF path

# Extract all tables from all pages
tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

# Combine all tables into one DataFrame (assuming table split across pages)
zip_county_df = pd.concat(tables, ignore_index=True)

# Preview the combined data
print(zip_county_df.head())

# --- Step 2: Clean ZIP-to-County DataFrame ---

# Standardize column names (strip spaces, lowercase)
zip_county_df.columns = zip_county_df.columns.str.strip().str.lower()

# You mentioned columns like 'zip', 'code city', 'county' — focus on 'zip' and 'county'
# Some columns might be messy; ensure 'zip' and 'county' columns exist
print(zip_county_df.columns)

# Convert ZIP codes to 5-digit strings (pad zeros if needed)
zip_county_df['zip'] = zip_county_df['zip'].astype(str).str.zfill(5)

# Strip whitespace in county names
zip_county_df['county'] = zip_county_df['county'].str.strip()

# Create mapping dictionary ZIP -> County
zip_to_county = dict(zip(zip_county_df['zip'], zip_county_df['county']))

# --- Step 3: Load your house price data ---
houses_df = pd.read_csv("socal2.csv")  # Replace with your CSV file path

# Ensure zip_code is a 5-digit string
houses_df['zip_code'] = houses_df['zip_code'].astype(str).str.zfill(5)

# Map ZIP codes to counties
houses_df['county'] = houses_df['zip_code'].map(zip_to_county)

# Remove rows where county couldn't be mapped
houses_df = houses_df.dropna(subset=['county'])

# --- Step 4: Calculate median house price per county ---
median_prices = houses_df.groupby('county')['price'].median().reset_index()

print("Median house prices by county:")
print(median_prices)
