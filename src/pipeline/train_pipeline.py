import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()
            logging.info("✅ Data ingestion completed.")
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:
            transformer = DataTransformation()
            X_train, y_train, X_test, y_test, preprocessor_path = transformer.initiate_data_transformation()
            logging.info(f"✅ Data transformation completed. Preprocessor saved at: {preprocessor_path}")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, X_train, y_train, X_test, y_test):
        try:
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            logging.info(f"✅ Model training completed. Model saved at: {model_path}")
            return model_path
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            print("🚀 Starting data ingestion...")
            self.start_data_ingestion()
            print("🔄 Starting data transformation...")
            X_train, y_train, X_test, y_test = self.start_data_transformation()
            print("🧠 Starting model training...")
            model_path = self.start_model_training(X_train, y_train, X_test, y_test)
            print("✅ Training pipeline completed. Model saved at:", model_path)
            logging.info(f"✅ Final model saved at: {model_path}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    TrainingPipeline().run_pipeline()
