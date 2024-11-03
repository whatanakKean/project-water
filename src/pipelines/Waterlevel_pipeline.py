import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

class WaterLevelExcelModel:
    def __init__(self, file_path, station_name='Bassac Chaktomuk', station_code=33401, look_back=10, batch_size=32):
        self.file_path = file_path
        self.station_name = station_name
        self.station_code = station_code
        self.look_back = look_back
        self.batch_size = batch_size
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.history = None

    def load_data(self):
        # Load Excel file without parsing dates
        data = pd.read_excel(
            self.file_path,
            sheet_name="in",
            skiprows=4,      
            header=[0, 1, 2]
        )
        # Fill NaN rainfall values with 0
        data.loc[:, (slice(None), slice(None), 'RF')] = data.loc[:, (slice(None), slice(None), 'RF')].fillna(0)
        data.set_index('StaName', inplace=True)
        
        # Filter for specified station measurements
        self.data = data.loc[:, pd.IndexSlice[self.station_name, self.station_code, :]]
        
        # Reset index and clean up columns
        self.data.reset_index(inplace=True)
        self.data.columns = self.data.columns.droplevel([0, 1])
        self.data.columns = ['Date', 'WL 7AM', 'WL 7PM', 'RF']
        
        # Process 'Date' and set as index
        self.data['Date'] = self.data['Date'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Interpolate missing values for WL columns
        self.data['WL 7AM'] = self.data['WL 7AM'].interpolate(method='linear')
        self.data.drop(['WL 7PM', 'RF'], axis=1, inplace=True)
    
    def preprocess_data(self, train_ratio=0.7, valid_ratio=0.15):
        # Split the data
        train_size = int(len(self.data) * train_ratio)
        valid_size = int(len(self.data) * valid_ratio)
        test_size = len(self.data) - train_size - valid_size
        
        train = self.data.iloc[:train_size]
        valid = self.data.iloc[train_size:train_size + valid_size]
        test = self.data.iloc[train_size + valid_size:]
        
        # Scale the data
        self.scaler.fit(train)  # Fit only on train data
        scaled_train = self.scaler.transform(train)
        scaled_valid = self.scaler.transform(valid)
        scaled_test = self.scaler.transform(test)
        
        # Create TimeSeriesGenerators
        self.train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=self.look_back, batch_size=self.batch_size)
        self.valid_generator = TimeseriesGenerator(scaled_valid, scaled_valid, length=self.look_back, batch_size=self.batch_size)
        self.test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=self.look_back, batch_size=self.batch_size)
        
    def build_model(self):
        # Define the LSTM-GRU model
        self.model = Sequential([
            LSTM(200, activation='relu', return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(100, return_sequences=True),
            GRU(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def train_model(self, epochs=100):
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )

    def evaluate_model(self):
        # Evaluate model on test data
        loss = self.model.evaluate(self.test_generator)
        print(f"Test Loss: {loss}")

    def save_model(self, model_path='water_level_model.keras', scaler_path='scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path='water_level_model.keras', scaler_path='scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def make_predictions(self, forward_days=5):
        if self.model is None:
            try:
                self.load_model()
            except:
                raise ValueError("Model is not loaded or built. Please train or load a model before making predictions.")
        
        # Generate predictions
        predictions = self.model.predict(self.test_generator)
        predictions = predictions.reshape(-1, 1)  # Reshape for scaler inverse transform
        predictions_inverse = self.scaler.inverse_transform(predictions)
        
        # Create future date range for predictions
        last_date = pd.to_datetime(self.data.index[-1])
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forward_days)
        
        # Prepare DataFrame for predictions
        results_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Water Level': predictions_inverse[-forward_days:].flatten()
        })
        
        return results_df

    def run_training_pipeline(self):
        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()

    def run_prediction_pipeline(self, forward_days=15):
        self.load_data()
        self.preprocess_data()
        return self.make_predictions(forward_days)


pipeline = WaterLevelExcelModel(file_path='./src/data/ManualData_Mainstream.xlsx', station_name='Bassac Chaktomuk', station_code=33401)
# pipeline.run_training_pipeline()
prediction = pipeline.run_prediction_pipeline(forward_days=15)
print(prediction)