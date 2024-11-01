import numpy as np
import pandas as pd
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import joblib
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
        df_non_zero = df_long[df_long['water_level'] != 0]
        df_non_zero.set_index('DATE_GMT', inplace=True)
        df_non_zero.index.freq='D'
        
        # Split the data into training and testing sets
        self.train = df_non_zero.iloc[:1222]
        self.test = df_non_zero.iloc[1222:]
        
        # Scale the data
        self.scaler.fit(self.train)  # Fit on train
        self.scaled_train = self.scaler.transform(self.train)
        self.scaled_test = self.scaler.transform(self.test)
        
        # Create TimeseriesGenerator for training and testing data
        self.train_generator = TimeseriesGenerator(self.scaled_train, self.scaled_train, length=self.look_back, batch_size=self.batch_size)
        self.test_generator = TimeseriesGenerator(self.scaled_test, self.scaled_test, length=self.look_back, batch_size=self.batch_size)

    def build_model(self):
        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(self.look_back, self.scaled_train.shape[1])))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(GRU(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(self.train_generator, epochs=100, callbacks=[early_stop], verbose=1)

    def evaluate_model(self):
        loss = self.model.evaluate(self.test_generator)
        print(f"Test Loss: {loss}")

    def save_model(self, model_path='lstm_model.keras', scaler_path='scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path='lstm_model.keras', scaler_path='scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def make_predictions(self, forward):
        # Check if model is loaded
        if self.model is None:
            self.load_model()

        # Make predictions
        predictions = self.model.predict(self.test_generator)

        # Reshape the predictions if needed (since TimeseriesGenerator may generate 3D data)
        predictions = predictions.reshape(-1, 1)

        # Inverse transform the predictions back to the original scale
        predictions_inverse = self.scaler.inverse_transform(predictions)

        # Get the last date from the test set DataFrame (df_non_zero) after preprocessing
        last_date = pd.to_datetime(self.test.index[-1])  # Access df_non_zero created in preprocess_data()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forward)

        # Prepare a DataFrame for future predictions
        results_df = pd.DataFrame({
            'timestamp': future_dates,
            'predicted': predictions_inverse[-forward:].flatten()  # Use the last 'forward' predictions
        })

        return results_df

    def run_training_pipeline(self):
        self.fetch_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()

    def run_prediction_pipeline(self, forward):
        self.fetch_data()
        self.preprocess_data()
        predictions_inverse = self.make_predictions(forward)
        return predictions_inverse
    

    # Get data
    def get_waterlevel(self, step='10D'):
        self.data = pd.read_csv('water_level.csv', index_col=0, parse_dates=True)
        waterlevel = self.data.last(step)
        return waterlevel
        


# # Example usage:
# pipeline = WaterLevelModel(station_code="PPB")
# pipeline.run_training_pipeline()

# pipeline = WaterLevelModel(station_code="PPB")
# prediction = pipeline.run_prediction_pipeline(forward=15)
# print(prediction.head(15))