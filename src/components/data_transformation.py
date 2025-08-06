import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler
from dataclasses import dataclass
from src.constant import TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataTransformationConfig:
    artifacts_dir: str = os.path.join('artifacts')
    ingested_train_path: str = os.path.join(artifacts_dir, 'phising.csv')
    transformed_train_file_path: str = os.path.join(artifacts_dir, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifacts_dir, 'test.npy')
    transformed_object_file_path: str = os.path.join(artifacts_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def initiate_data_transformation(self):
        logging.info("🚀 Starting data transformation...")

        try:
            # Load data
            df = pd.read_csv(self.config.ingested_train_path)
            logging.info(f"✅ Loaded data from {self.config.ingested_train_path}")

            # Clean object columns
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df.replace('?', np.nan, inplace=True)
            logging.info("🧹 Cleaned data: stripped spaces and replaced '?' with NaN")

            # Separate features and target
            X = df.drop(columns=TARGET_COLUMN)
            y = np.where(df[TARGET_COLUMN] == -1, 0, 1)
            logging.info("📦 Separated X and y")

            # Balance classes
            sampler = RandomOverSampler()
            X_sampled, y_sampled = sampler.fit_resample(X, y)
            logging.info("⚖️ Applied RandomOverSampler to balance classes")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
            logging.info(f"✂️ Split data: Train shape {X_train.shape}, Test shape {X_test.shape}")

            # Build preprocessing pipeline
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", RobustScaler())
            ])
            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)
            logging.info("🔧 Applied imputation and scaling")

            # Save transformed arrays
            np.save(self.config.transformed_train_file_path, {"X": X_train_transformed, "y": y_train})
            np.save(self.config.transformed_test_file_path, {"X": X_test_transformed, "y": y_test})
            logging.info("💾 Saved transformed train and test data as .npy")

            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.transformed_object_file_path), exist_ok=True)
            self.utils.save_object(self.config.transformed_object_file_path, pipeline)
            logging.info(f"📦 Saved preprocessor at {self.config.transformed_object_file_path}")

            logging.info("✅ Data transformation completed successfully.")
            return (
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test,
                self.config.transformed_object_file_path
            )

        except Exception as e:
            logging.error("❌ Error during data transformation")
            raise CustomException(e, sys) from e
