import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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
        self.filepath = f"../models/LSTM_{station_code}.keras"
    
    def prepare_data(self):
        self.train = self.data.iloc[:1212]
        self.test = self.data.iloc[1212:]
        
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
        print("Scaled test: ", len(self.scaled_test))
        

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
            self.model = tf.keras.models.load_model(self.filepath)

        # Make predictions
        predictions = self.model.predict(self.test_generator)

        # Inverse scale predictions to original values
        predictions_inverse = self.scaler.inverse_transform(predictions)

        # Create a DataFrame for predictions
        prediction_dates = self.test.index[self.look_back:]
        predictions_df = pd.DataFrame(
            data=predictions_inverse, 
            index=prediction_dates, 
            columns=['Predicted_Water_Level']
        )

        # Add predictions to the test DataFrame
        self.test = self.test.copy()  # Ensure we don't modify the original DataFrame
        self.test['Predicted_Water_Level'] = predictions_df['Predicted_Water_Level']
        #Exclude 10 days
        self.test = self.test.iloc[11:]

        # Optionally, save the updated DataFrame to a CSV file
        self.test.to_csv(f'../data/data_predicted_PPB.csv')

        # Merge prediction to original data 
        
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
