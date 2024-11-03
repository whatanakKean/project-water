import numpy as np
import pandas as pd
import requests
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

LOWER_MEKONG_STATION_CODES =  [
    "STR", # StungTreng
    "KRA", # Kratie
    "KOM", # Kompong Cham
    "PPB", # Phnom Penh (Bassac)
    "PPP", # Phnom Penh Port
    "KOH", # Koh Khel (Bassac)
    "NEA", # Neak Luong
    "PRE", # Prek Kdam (Tonle Sap)
    "TCH", # Tan Chau
    "CDO", # Chau Doc (Bassac)
]

class WaterLevelModel:
    def __init__(self, station_code="PPB", look_back=15, batch_size=32):
        self.station_code = station_code
        self.look_back = look_back
        self.batch_size = batch_size
        self.BASE_URL = "http://ffw.mrcmekong.org/fetchwet_st.php?StCode="
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.df_non_zero = None  # Initialize df_non_zero as an instance variable

    def fetch_data(self):
        response = requests.get(self.BASE_URL + self.station_code, verify=False)
        data_string = response.content.decode('utf-8')

        # Convert single quotes and remove any non-JSON parts
        data_string = data_string.replace('date_gmt:', '"date_gmt":')
        data_string = data_string.replace('Max:', '"Max":')
        data_string = data_string.replace('Min:', '"Min":')
        data_string = data_string.replace('AVG:', '"AVG":')
        data_string = data_string.replace('floodLevel:', '"floodLevel":')
        data_string = data_string.replace('alarmLevel:', '"alarmLevel":')
        
        for year in range(1992, 2025):
            data_string = data_string.replace(f'{year}:', f'"{year}":')
        data_string = data_string.replace(',]', ']')

        # Now parse it into a list of dictionaries
        self.data = json.loads(data_string)

    def preprocess_data(self):
        df = pd.DataFrame(self.data)
        df['date_gmt'] = df['date_gmt'].apply(lambda x: x.split("-")[1] + "-" + x.split("-")[2])
        df['station'] = self.station_code
        
        # Set date_gmt as index 
        df.index = df['date_gmt']

        # Filter relevant columns
        df_filtered = df[['date_gmt', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']]
        df_filtered.set_index('date_gmt', inplace=True)
        df_filtered.reset_index(inplace=True)
        df_long = pd.melt(df_filtered, id_vars=['date_gmt'], value_vars=['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
                        var_name='Year', value_name='water_level')
        df_long['DATE_GMT'] = pd.to_datetime(df_long['date_gmt'] + '-' + df_long['Year'], format='%m-%d-%Y').dt.strftime('%m-%d-%Y')
        df_long = df_long[['DATE_GMT', 'water_level']]
        self.df_non_zero = df_long[df_long['water_level'] != 0]  # Store as instance variable
        self.df_non_zero.set_index('DATE_GMT', inplace=True)
        self.df_non_zero.index.freq='D'
        
        # Scale the data
        self.scaler.fit(self.df_non_zero)  # Fit on the non-zero data
        self.scaled_data = self.scaler.transform(self.df_non_zero)

        # Create TimeseriesGenerator for the data
        self.data_generator = TimeseriesGenerator(self.scaled_data, self.scaled_data, length=self.look_back, batch_size=self.batch_size)

    def load_model(self, model_path='water_level_model.keras', scaler_path='scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def make_predictions(self, forward):
        # Check if model is loaded
        if self.model is None:
            self.load_model()

        # Make predictions
        predictions = self.model.predict(self.data_generator)

        # Reshape the predictions if needed
        predictions = predictions.reshape(-1, 1)

        # Inverse transform the predictions back to the original scale
        predictions_inverse = self.scaler.inverse_transform(predictions)

        # Get the last date from the processed DataFrame
        last_date = pd.to_datetime(self.df_non_zero.index[-1])  # Access df_non_zero from instance variable
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forward)

        # Prepare a DataFrame for future predictions
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted': predictions_inverse[-forward:].flatten()  # Use the last 'forward' predictions
        })

        return results_df

    def run_prediction_pipeline(self, forward):
        self.fetch_data()
        self.preprocess_data()
        predictions_inverse = self.make_predictions(forward)
        return predictions_inverse


# Example usage:
pipeline = WaterLevelModel(station_code="PPB")
prediction = pipeline.run_prediction_pipeline(forward=3)
print(prediction.head(5))
