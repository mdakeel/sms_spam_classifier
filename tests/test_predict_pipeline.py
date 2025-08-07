import sys
import os

# 🔧 Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == '__main__':
    print('🚀 Testing Batch Prediction Pipeline...')
    pipeline = PredictPipeline()
    result = pipeline.run_pipeline(["Congratulations! You've won ₹50,000. Click here to claim.", "Hey, are we still meeting today?", "Your account has been suspended. Verify now: http://fakebank.in"])
    print(result)  # ['spam', 'ham']
    print('\n🎯 Batch prediction pipeline test completed successfully.')

