import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.model_trainer import ModelTrainer
from src.utils.main_utils import MainUtils

if __name__ == '__main__':
    # Load processed data
    train_data = dict(np.load('artifacts/train.npz', allow_pickle=True))
    test_data = dict(np.load('artifacts/test.npz', allow_pickle=True))

    x_train, y_train = train_data['X'], train_data['y']
    x_test, y_test = test_data['X'], test_data['y']

    # Optional slicing for quick tests
    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    # Train and save model
    trainer = ModelTrainer()
    model_path = trainer.initiate_model_trainer(x_train, y_train, x_test, y_test)
    print(f"✅ Model saved at: {model_path}")

    # 🔍 Sanity check: Load and predict
    utils = MainUtils()
    model = utils.load_object(model_path)
    sample_preds = model.predict(x_test[:5])
    print("🔍 Sample predictions:", sample_preds)
