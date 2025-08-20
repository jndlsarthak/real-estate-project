import pandas as pd
from geopy.geocoders import MapBox
from geopy.distance import geodesic
from dotenv import load_dotenv
import os

# Loading the environment variables from .env file
load_dotenv()

# Getting the Mapbox API key from environment variable
mapbox_token = os.getenv('MAPBOX_API_KEY')

if not mapbox_token:
    raise ValueError("Mapbox API key not found. Please set it in the .env file.")

# Initializing MapBox geocoder with the API token
geolocator = MapBox(api_key=mapbox_token)

# City centre coordinates for Magnitogorsk (latitude, longitude)
city_centre_coords = (53.4186, 59.0472)

# Defining an approximate bounding box for Magnitogorsk city
def is_in_city(latitude, longitude):
    return 53.3 <= latitude <= 53.55 and 58.9 <= longitude <= 59.2

def get_distance_and_city_flag(address):
    try:
        full_address = f"{address.strip()}, Magnitogorsk, Russia"
        location = geolocator.geocode(full_address)
        if location:
            lat, lon = location.latitude, location.longitude
            print(f"Geocoded '{full_address}' to: ({lat}, {lon})")
            in_city_flag = is_in_city(lat, lon)
            if in_city_flag :
                distance_km = geodesic(city_centre_coords, (lat, lon)).km  
                return distance_km, in_city_flag
            else :
                return None, False
        else:
            print(f"Failed to geocode '{full_address}'")
            return None, False
    except Exception as e:
        print(f"Error geocoding '{address}': {e}")
        return None, False
    

# Loading the data from CSV 
df = pd.read_excel('../data/russianREdata.xlsx', sheet_name='Продажа квартир',header=4)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]# Dropping unnamed/empty columns
df = df.dropna(subset=['Цена, р.'])# Dropping the row with missing target value

# Applying function and split results into two new columns
df[['distance_from_city_centre_km', 'in_city']] = df["Адрес"].apply(lambda x: pd.Series(get_distance_and_city_flag(x)))


# Calculating 75th percentile distance from entries where in_city is True and distance is not None
valid_distances = df.loc[(df['in_city'] == True) & (df['distance_from_city_centre_km'].notnull()), 'distance_from_city_centre_km']
median_75 = valid_distances.quantile(0.75)

# Replacing distances and flags where in_city == False or distance is None
mask = (df['in_city'] == False) | (df['distance_from_city_centre_km'].isnull())
df.loc[mask, 'distance_from_city_centre_km'] = median_75
df.loc[mask, 'in_city'] = "Don't know"

# Selecting only the two new columns
new_df = df[['Дата','Цена, р.', 'distance_from_city_centre_km', 'in_city']]
new_df = new_df.rename(columns={'Дата': 'Date', 'Цена, р.':'Price, RUR'})

# Saving to a new CSV file
new_df.to_csv('../data/distance_and_city_flag_only.csv', index=False)
print("New CSV file created")


#Merging the two files to create a new file with all the original data and the new distance and in_city feature 
older_df = pd.read_csv('../data/final01_processed_real_estate_data.csv') 
newer_df = pd.read_csv('../data/distance_and_city_flag_only.csv')
older_df = older_df.reset_index(drop=True)
newer_df = newer_df.reset_index(drop=True)

# Concatenating desired columns from new_df to old_df by index
merged_df = pd.concat([older_df, newer_df[['distance_from_city_centre_km', 'in_city']]], axis=1)

merged_df[' Date'] = newer_df['Date']

# Saving merged DataFrame to CSV
merged_df.to_csv('../data/merged_output.csv', index=False)

print("Merged file created.")
