import os
import sys


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    '''
    This class is responsible for defining the configuration for data ingestion.'''
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # Check if the directory exists, if not create it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test set to respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
        
if __name__ == "__main__":
    # This block is used to test the DataIngestion class when the script is run directly.
    # It will create an instance of the DataIngestion class and call the initiate_data_ingestion method.
    obj = DataIngestion()
    train_data, test_data =  obj.initiate_data_ingestion()
    print("Data Ingestion completed successfully.")
    
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    print("Data Transformation completed successfully.")
    