from flask import Flask, render_template, request, send_file
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/csv_check', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        pipeline = PredictionPipeline(request)
        try:
            output = pipeline.run_pipeline()
            return send_file(output.prediction_file_path, as_attachment=True)
        except Exception as e:
            return f"<h3>Error: {e}</h3>"
    return render_template('csv_check.html')

@app.route('/url-check', methods=['GET', 'POST'])
def url_check():
    if request.method == 'POST':
        url = request.form['website_url']
        try:
            pipeline = PredictionPipeline()
            result = pipeline.predict_from_url(url)
            return render_template('url_result.html', url=url, result=result)
        except Exception as e:
            return f"<h3>Error: {e}</h3>"
    return render_template('url_check.html')

if __name__ == '__main__':
    app.run(debug=True)
