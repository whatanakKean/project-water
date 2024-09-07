import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

class LSTMModelTrainer:
    def __init__(self, data, station_code, look_back=10, batch_size=32):
        self.data = data
        self.look_back = look_back
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.filepath = f"../models/LSTM_{station_code}.h5"
    
    def prepare_data(self):
        self.train = self.data.iloc[:1222]
        self.test = self.data.iloc[1222:]
        
        # Scale the data
        self.scaled_train = self.scaler.fit_transform(self.train)
        self.scaled_test = self.scaler.transform(self.test)
        
        # Create TimeseriesGenerator for training and testing data
        self.train_generator = TimeseriesGenerator(
            self.scaled_train, self.scaled_train, 
            length=self.look_back, batch_size=self.batch_size
        )
        self.test_generator = TimeseriesGenerator(
            self.scaled_test, self.scaled_test, 
            length=self.look_back, batch_size=self.batch_size
        )

    def train_model(self, epochs=100, patience=10):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.look_back, self.scaled_train.shape[1])))
        self.model.add(Dense(self.scaled_train.shape[1]))
        self.model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        self.model.save(self.filepath)
        
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
        - loss: The loss value on the test set.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call 'build_model()' first.")
        
        loss = self.model.evaluate(self.test_generator, verbose=0)
        return loss

    def predict(self, data):
        """
        Make predictions with the trained model.
        
        Parameters:
        - data: DataFrame with new data for prediction.
        
        Returns:
        - predictions: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call 'build_model()' first.")
        
        # Scale the input data
        scaled_data = self.scaler.transform(data)
        
        # Create TimeseriesGenerator for new data
        generator = TimeseriesGenerator(
            scaled_data, scaled_data,
            length=self.look_back, batch_size=1
        )
        
        predictions = []
        for i in range(len(generator)):
            x, _ = generator[i]
            y_pred = self.model.predict(x)
            predictions.append(y_pred[0])
        
        # Inverse transform the predictions
        predictions = self.scaler.inverse_transform(np.array(predictions))
        
        return predictions
