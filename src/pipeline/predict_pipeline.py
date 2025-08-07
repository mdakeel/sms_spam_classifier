import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        self.transformed = DataTransformation()
        self.utils = MainUtils()
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
    def load_artifacts(self):
        try:
            logging.info(f"🔍 Loading model from: {self.model_path}")
            logging.info(f"🔍 Loading preprocessor from: {self.preprocessor_path}")
            
            model = self.utils.load_object(self.model_path)
            preprocessor = self.utils.load_object(self.preprocessor_path)
            
            logging.info("✅ Artifacts loaded successfully.")
            return model, preprocessor
        except Exception as e:
            logging.error(f"❌ Failed to load artifacts: {e}", exc_info=True)
            raise CustomException("Error loading model or preprocessor.", sys) from e
        
    def preprocess_input(self, message, preprocessor):
        try:
            logging.info("🔄 Preprocessing input message...")
            
            # Preprocess each message
            transformed = [" ".join(self.transformed.transform_text(msg)) for msg in message] 
            logging.info("🔄 Message Seccessfully Transformed.")
            
            #vectorization
            vectorizer = preprocessor.transform(transformed)
            logging.info("✅ Message vectorized successfully.")
            return vectorizer
        except Exception as e:
            logging.error(f"❌ Error in preprocessing message: {e}", exc_info=True)
            raise CustomException(e, sys)
        
    def predict_from_dic(self, vectorizer):
        try:
            model, _ = self.load_artifacts()
            preds = model.predict(vectorizer)

            return ['spam' if p == 1 else 'ham' for p in preds]      
        except Exception as e:
             raise CustomException(e, sys)
         
    def run_pipeline(self, messages):
        try:
            model, preprocessor = self.load_artifacts()
            X = self.preprocess_input(messages, preprocessor)
            preds = model.predict(X)
            return ['spam' if p == 1 else 'ham' for p in preds]
        except Exception as e:
            raise CustomException(e, sys)

