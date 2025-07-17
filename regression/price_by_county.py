import pandas as pd

# Load your Excel file with ZIP to County info
zip_county_df = pd.read_excel("zip_county.xlsx")

# Take a quick look at the columns and some rows to verify
print(zip_county_df.head())

# Standardize column names to lowercase and strip spaces
zip_county_df.columns = zip_county_df.columns.str.strip().str.lower()

# Sometimes columns may have spaces, e.g. 'code city county'
# From your snippet, looks like columns: 'zip', 'code city', 'county'
# You may want to rename for clarity, or just use 'zip' and 'county' columns:

# If your zip codes are stored as integers, convert to strings with leading zeros if needed
zip_county_df['zip'] = zip_county_df['zip'].astype(str).str.zfill(5)

# Similarly clean county column (strip whitespace)
zip_county_df['county'] = zip_county_df['county'].str.strip()

# Create a dictionary for fast lookup
zip_to_county = dict(zip(zip_county_df['zip'], zip_county_df['county']))

# Now load your house prices CSV or DataFrame (example)
houses_df = pd.read_csv("houses.csv")

# Convert zip_code in your houses data to string and pad zeros if necessary
houses_df['zip_code'] = houses_df['zip_code'].astype(str).str.zfill(5)

# Map zip codes to counties using your loaded dictionary
houses_df['county'] = houses_df['zip_code'].map(zip_to_county)

# Drop houses with unknown counties if needed
houses_df = houses_df.dropna(subset=['county'])

# Calculate median house price per county
median_prices = houses_df.groupby('county')['price'].median().reset_index()

print(median_prices)
