import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import NetworkSecurityException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started data ingestion process.")
        try:
            # Load your main dataset here
            train_df = pd.read_excel('notebook\data\TRAIN_NEW\TRAIN_CATEGORICAL_METADATA_new.xlsx')
            fmri = pd.read_csv('notebook\data\TRAIN_NEW\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
            quant_df = pd.read_excel('notebook\data\TRAIN_NEW\TRAIN_QUANTITATIVE_METADATA_new.xlsx')
            train_sols = pd.read_excel('notebook\data\TRAIN_NEW\TRAINING_SOLUTIONS.xlsx')

            train_df = train_df.merge(fmri,on='participant_id',how = 'left')
            train_df = train_df.merge(quant_df,on='participant_id',how = 'left')
            train_df = train_df.merge(train_sols,on='participant_id',how = 'left')


            train_df.head()

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Dataset loaded successfully.")
            train_df.drop(columns='participant_id',axis=1, inplace=True)

            train_set, test_set = train_test_split(train_df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train-test split done and saved.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise NetworkSecurityException(e, sys)
