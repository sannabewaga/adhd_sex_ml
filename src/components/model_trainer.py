import os
import sys
import pandas as pd  # ✅ Missing import
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.exception import NetworkSecurityException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


@dataclass
class ModelTrainingConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array,preprocessor_path):
        try:
            logging.info('Splitting training and test data')

            # Assuming ADHD_Outcome is the target and the rest are features
            Xtrain = train_array[:, :-2]  # All columns except the last two (features)
            Xtest = test_array[:, :-2]    # All columns except the last two (features)

            ytrain_adhd = train_array[:, -2]  # Last but one column (ADHD_Outcome)
            ytest_adhd = test_array[:, -2]   # Last but one column (ADHD_Outcome)

            ytrain_sex = train_array[:, -1]  # Last column (Sex_F)
            ytest_sex = test_array[:, -1]   # Last column (Sex_F)

            # Create and configure Logistic Regression model
            model = LogisticRegression(class_weight='balanced', max_iter=1000)

            # Train the model on the ADHD outcome
            model.fit(Xtrain, ytrain_adhd)

            # Make predictions on the test set
            ypred_adhd = model.predict(Xtest)

            # Evaluate the model
            logging.info("Model training and evaluation completed successfully.")
            logging.info(f"Accuracy: {accuracy_score(ytest_adhd, ypred_adhd)}")
            logging.info(f"Confusion Matrix: \n{confusion_matrix(ytest_adhd, ypred_adhd)}")
            logging.info(f"Classification Report: \n{classification_report(ytest_adhd, ypred_adhd)}")

            # Save the trained model to the specified file path
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)

            return model

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

