import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataValidationConfig:
    validation_report_path: str = os.path.join('artifacts', 'validation_report.csv')

class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()

    def validate_data(self, X_train, y_train, X_test, y_test):
        logging.info("Starting data validation...")

        try:
            # Shape checks
            assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train"
            assert X_test.shape[0] == y_test.shape[0], "Mismatch in X_test and y_test"
            assert X_train.ndim == 2 and X_test.ndim == 2, "X must be 2D"
            assert y_train.ndim == 1 and y_test.ndim == 1, "y must be 1D"

            # Missing values
            missing_train = np.isnan(X_train).sum()
            missing_test = np.isnan(X_test).sum()

            # Class balance
            train_dist = dict(Counter(y_train))
            test_dist = dict(Counter(y_test))

            # Numeric check
            is_numeric = np.issubdtype(X_train.dtype, np.number)

            # Save report
            report = {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "Missing_X_train": missing_train,
                "Missing_X_test": missing_test,
                "Class_distribution_train": train_dist,
                "Class_distribution_test": test_dist,
                "Is_numeric": is_numeric
            }

            pd.DataFrame([report]).to_csv(self.config.validation_report_path, index=False)
            logging.info(f"Validation report saved at: {self.config.validation_report_path}")
            logging.info("Data validation completed successfully.")
            return True

        except Exception as e:
            logging.info("Error during data validation")
            raise CustomException(e, sys) from e
