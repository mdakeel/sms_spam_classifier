import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    transformer = DataTransformation()
    print("Starting data transformation process...")
    
    X_train, y_train, X_test, y_test, preprocessor_path = transformer.initiate_data_transformation()

    print(f"Transformation Object saved at: {preprocessor_path}")
    print("Data Transformation process finished.")
