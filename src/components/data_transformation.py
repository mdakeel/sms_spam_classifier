import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from src.constant import TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import joblib

@dataclass
class DataTransformationConfig:
    artifacts_dir: str = os.path.join('artifacts')
    ingested_train_path: str = os.path.join(artifacts_dir, 'sms.csv')
    transformed_train_file_path: str = os.path.join(artifacts_dir, 'train.npz')
    transformed_test_file_path: str = os.path.join(artifacts_dir, 'test.npz')
    transformed_object_file_path: str = os.path.join(artifacts_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()
        self.tokenizer = TreebankWordTokenizer()
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def transform_text(self, text):
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        tokens = [i for i in tokens if i.isalnum()]
        tokens = [i for i in tokens if i not in self.stop_words and i not in string.punctuation]
        tokens = [self.ps.stem(i) for i in tokens]
        return tokens

    def initiate_data_transformation(self):
        logging.info("🚀 Starting data transformation...")

        try:
            df = pd.read_csv(self.config.ingested_train_path)
            logging.info(f"✅ Loaded data from {self.config.ingested_train_path}")

            # Drop unnecessary columns if they exist
            for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            logging.info("✅ Removed unnecessary columns")

            # Encode target
            encoder = LabelEncoder()
            df['target'] = encoder.fit_transform(df[TARGET_COLUMN])
            logging.info("✅ Label encoding done")

            # Remove duplicates
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info("✅ Removed duplicates")

            # Feature engineering
            df['num_characters'] = df['Message'].apply(len)
            df['num_words'] = df['Message'].apply(lambda x: len(self.tokenizer.tokenize(x)))
            df['transformed_text'] = df['Message'].apply(self.transform_text)
            df['joined_text'] = df['transformed_text'].apply(lambda x: ' '.join(x))

            # Vectorization
            tfidf = TfidfVectorizer(max_features=3000)
            X_vectors = tfidf.fit_transform(df['joined_text']).toarray()
            y = df['target']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)
            logging.info(f"✂️ Split data: Train shape {X_train.shape}, Test shape {X_test.shape}")

            # Save transformed arrays
            np.savez(self.config.transformed_train_file_path, X=X_train, y=y_train)
            np.savez(self.config.transformed_test_file_path, X=X_test, y=y_test)
            logging.info("💾 Saved transformed train and test data")

            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.transformed_object_file_path), exist_ok=True)
            joblib.dump(tfidf, self.config.transformed_object_file_path)
            logging.info(f"📦 Saved preprocessor at {self.config.transformed_object_file_path}")

            logging.info("✅ Data transformation completed successfully.")
            return X_train, y_train, X_test, y_test, self.config.transformed_object_file_path

        except Exception as e:
            logging.error("❌ Error during data transformation")
            raise CustomException(e, sys) from e
