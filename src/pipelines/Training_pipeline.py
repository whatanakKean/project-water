import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.Data_ingestion import DataIngestion
from components.Data_preprocessing import DataPreprocessor
from components.Model_training import LSTMModelTrainer

station_code = "PPB"
mekong_data = DataIngestion(station_code)
# df = mekong_data.get_data()
df = mekong_data.get_data_local()

preprocessor = DataPreprocessor(df)
df_preprocessed = preprocessor.preprocess_data()

trainer = LSTMModelTrainer(df_preprocessed, station_code)
trainer.prepare_data()
# trainer.train_model()
trainer.evaluate_model()