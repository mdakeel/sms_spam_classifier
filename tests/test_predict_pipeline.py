import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.predict_pipeline import PredictionPipeline

if __name__ == '__main__':
    print('🚀 Testing Batch Prediction Pipeline...')

    # ✅ Replace with your actual test CSV path
    test_csv_path = 'notebooks/data/phising.csv'

    try:
        # Run pipeline directly from CSV path
        pipeline = PredictionPipeline(request=None)  # No Flask request needed
        output_detail = pipeline.get_predicted_dataframe(test_csv_path)

        print(f'✅ Prediction file created at: {pipeline.prediction_file_detail.prediction_file_path}')
        print('🎯 Batch prediction pipeline test completed successfully.')

    except Exception as e:
        print(f'❌ Error during prediction: {e}')
