import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model", "model.pkl")
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join("config", "model.yaml")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            "LogisticRegression": LogisticRegression(),
            "GaussianNB": GaussianNB(),
            "XGBClassifier": XGBClassifier()
        }
        self.model_param_grid = self.utils.read_yaml_file(self.config.model_config_file_path)["model_selection"]["model"]

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        logging.info("Evaluating base models...")
        report = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score
            logging.info(f"{name} Accuracy: {score:.4f}")
        return report

    def finetune_best_model(self, model_name, model, X_train, y_train):
        logging.info(f"Fine-tuning {model_name}...")
        param_grid = self.model_param_grid[model_name]["search_param_grid"]
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        logging.info(f"Best params for {model_name}: {best_params}")
        model.set_params(**best_params)
        return model

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test)
            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]
            logging.info(f"Best base model: {best_model_name} with accuracy {model_report[best_model_name]:.4f}")

            best_model = self.finetune_best_model(best_model_name, best_model, X_train, y_train)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            final_score = accuracy_score(y_test, y_pred)
            logging.info(f"Final accuracy after tuning: {final_score:.4f}")

            if final_score < self.config.expected_accuracy:
                raise CustomException(f"Model accuracy {final_score:.4f} below threshold {self.config.expected_accuracy}")

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            self.utils.save_object(self.config.trained_model_path, best_model)
            logging.info(f"Model saved at: {self.config.trained_model_path}")
            return self.config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
