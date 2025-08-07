import sys
import os
import numpy as np
from dataclasses import dataclass
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    expected_accuracy: float = 0.45

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            "GaussianNB": GaussianNB(),
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB()
        }

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        logging.info("🔍 Evaluating base models...")
        report = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            report[name] = {
                "accuracy": acc,
                "precision": prec
            }
            logging.info(f"{name} → Accuracy: {acc:.4f}, Precision: {prec:.4f}")
        return report

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test)
            best_model_name = max(model_report, key=lambda k: model_report[k]["precision"])
            best_model = self.models[best_model_name]
            logging.info(f"🏆 Best model: {best_model_name} with accuracy {model_report[best_model_name]['precision']:.4f}")

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            final_score_acc = accuracy_score(y_test, y_pred)
            final_score_prec = precision_score(y_test, y_pred)
            logging.info(f"✅ Final Accuracy: {final_score_acc:.4f}")
            logging.info(f"✅ Final Precision: {final_score_prec:.4f}")

            if final_score_acc < self.config.expected_accuracy:
                raise CustomException(f"Model accuracy {final_score_acc:.4f} below threshold {self.config.expected_accuracy}")

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            self.utils.save_object(self.config.trained_model_path, best_model)
            logging.info(f"📦 Model saved at: {self.config.trained_model_path}")
            return self.config.trained_model_path

        except Exception as e:
            logging.error("❌ Error during model training")
            raise CustomException(e, sys) from e
