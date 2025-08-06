import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.constant import PHISING_DATA_PATH
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    artifacts_folder: str = 'artifacts'
    phising_data = 'phising.csv'
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started...")
        try:
            #creating a folder 
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f"Artifacts folder created at: {self.config.artifacts_folder}")
            
            #extracting path
            destination_path = os.path.join(self.config.artifacts_folder, self.config.phising_data)
            logging.info(f'Copying data from {PHISING_DATA_PATH}')
            
            #reading data and sending to destination path
            df = pd.read_csv(PHISING_DATA_PATH)
            df.to_csv(destination_path, index=False)
            logging.info(f"Data saved to {destination_path}")
            
            logging.info("Data ingestion completed successfully.")
        except Exception as e:
            logging.info(f"Error occured during data ingestion: {str(e)}")
            raise CustomException(e, sys) from e
