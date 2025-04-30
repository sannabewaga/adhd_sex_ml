import os
import sys
import pandas as pd
import joblib

from src.exception import NetworkSecurityException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

class PredictPipeline:
    def __init__(self):
        try:
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.adhd_model_path = os.path.join("artifacts", "adhd_model.pkl")
            self.sex_model_path = os.path.join("artifacts", "sex_model.pkl")

            self.preprocessor = joblib.load(self.preprocessor_path)
            self.adhd_model = joblib.load(self.adhd_model_path)
            self.sex_model = joblib.load(self.sex_model_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_from_test_data(self):
        try:
            # Step 1: Ingest test data
            ingestion = DataIngestion()
            _, test_path = ingestion.initiate_data_ingestion()

            test_df = pd.read_csv(test_path)

            # Step 2: Drop target columns if they exist (safe handling)
            test_input = test_df.drop(columns=[col for col in ['ADHD_Outcome', 'Sex_F'] if col in test_df.columns])

            # Step 3: Transform
            transformed_data = self.preprocessor.transform(test_input)

            # Step 4: Predict
            adhd_pred = self.adhd_model.predict(transformed_data)
            sex_pred = self.sex_model.predict(transformed_data)

            # Step 5: Output
            result = pd.DataFrame({
                'participant_id': test_df['participant_id'],
                'ADHD_Prediction': adhd_pred,
                'Sex_Prediction': sex_pred
            })

            logging.info("Predictions generated successfully.")
            return result

        except Exception as e:
            raise NetworkSecurityException(e, sys)
