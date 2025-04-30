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

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transfomer_object(self, num_columns):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('num cols completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self, train_file_path, test_file_path):
        try:
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            target_col_adhd = 'ADHD_Outcome'
            target_col_sex = 'Sex_F'

            input_feature_train_df = train_df.drop(columns=[target_col_adhd, target_col_sex], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_col_adhd, target_col_sex], axis=1)

            target_adhd_train = train_df[target_col_adhd]
            target_sex_train = train_df[target_col_sex]

            target_adhd_test = test_df[target_col_adhd]
            target_sex_test = test_df[target_col_sex]

            preprocessing_obj = self.get_data_transfomer_object(input_feature_train_df.columns)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_adhd_train), np.array(target_sex_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_adhd_test), np.array(target_sex_test)
            ]

            logging.info("Data transformation completed successfully.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)



if __name__ == '__main__':
    obj = DataIngestion()
    train,test = obj.initiate_data_ingestion()

    data_trans = DataTransformation()
    data_trans.initiate_data_transformation(train,test)
    
