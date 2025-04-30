import os
import sys
import pandas as pd  # ✅ Missing import
import numpy as np

from dataclasses import dataclass

from src.exception import NetworkSecurityException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test data')

            # Assuming ADHD_Outcome is the target and the rest are features
            Xtrain = train_arr[:, :-2]  # All columns except the last two (features)
            Xtest = test_arr[:, :-2]    # All columns except the last two (features)

            ytrain_adhd = train_arr[:, -2]  # Last but one column (ADHD_Outcome)
            ytest_adhd = test_arr[:, -2]   # Last but one column (ADHD_Outcome)

            ytrain_sex = train_arr[:, -1]  # Last column (Sex_F)
            ytest_sex = test_arr[:, -1]   # Last column (Sex_F)

            # Create and configure Logistic Regression model
            model_adhd = LogisticRegression(class_weight='balanced', max_iter=1000)
            model_sex = LogisticRegression(class_weight='balanced', max_iter=1000)

            # Train the model on the ADHD outcome
            model_adhd.fit(Xtrain, ytrain_adhd)

            # Train the model for Sex_F
            model_sex.fit(Xtrain, ytrain_sex)

            # Make predictions on the test set for both targets
            ypred_adhd = model_adhd.predict(Xtest)
            ypred_sex = model_sex.predict(Xtest)

            # Evaluate both models
            logging.info("Model training and evaluation completed successfully.")

            # Evaluate ADHD model
            logging.info(f"ADHD Model Accuracy: {accuracy_score(ytest_adhd, ypred_adhd)}")
            logging.info(f"ADHD Model Confusion Matrix: \n{confusion_matrix(ytest_adhd, ypred_adhd)}")
            logging.info(f"ADHD Model Classification Report: \n{classification_report(ytest_adhd, ypred_adhd)}")

            # Evaluate Sex model
            logging.info(f"Sex Model Accuracy: {accuracy_score(ytest_sex, ypred_sex)}")
            logging.info(f"Sex Model Confusion Matrix: \n{confusion_matrix(ytest_sex, ypred_sex)}")
            logging.info(f"Sex Model Classification Report: \n{classification_report(ytest_sex, ypred_sex)}")

            # Save both trained models to the specified file path
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj={'adhd_model': model_adhd, 'sex_model': model_sex})

            return model_adhd, model_sex

        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train,test = obj.initiate_data_ingestion()

    data_trans = DataTransformation()
    train_array,test_array,preprocessor_file_path = data_trans.initiate_data_transformation(train,test)

    trainer = ModelTrainer()
    model1,model2 = trainer.initiate_model_training(train_arr=train_array,test_arr=test_array)

    
