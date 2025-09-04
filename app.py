#!/usr/bin/env python3
"""
Healthcare Sales Analysis & Forecasting Flask Backend
Main application file with API endpoints
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Import custom modules
from models.forecasting import ForecastingEngine
from models.analytics import AnalyticsEngine
from utils.data_processor import DataProcessor
from utils.insights import InsightsGenerator
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for API calls

# Initialize engines
forecasting_engine = ForecastingEngine()
analytics_engine = AnalyticsEngine()
data_processor = DataProcessor()
insights_generator = InsightsGenerator()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data store (in production, use Redis or database)
current_data = None
processed_data = None

@app.route('/')
def index():
    """Render main dashboard page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial processing"""
    global current_data, processed_data
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if filename.endswith('.csv'):
                    current_data = pd.read_csv(filepath)
                else:
                    current_data = pd.read_excel(filepath)
                
                processed_data = data_processor.process_data(current_data)
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully',
                    'rows': len(processed_data),
                    'columns': list(processed_data.columns)
                })
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return jsonify({'error': f'Error processing file: {str(e)}'}), 400
        
        else:
            return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed.'}), 400
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error during upload'}), 500

@app.route('/api/demo-data', methods=['GET'])
def generate_demo_data():
    """Generate synthetic pharmaceutical sales data"""
    global current_data, processed_data
    
    try:
        demo_data = data_processor.generate_demo_data()
        current_data = demo_data
        processed_data = data_processor.process_data(demo_data)
        
        return jsonify({
            'success': True,
            'message': 'Demo data generated successfully',
            'rows': len(processed_data),
            'columns': list(processed_data.columns)
        })
    
    except Exception as e:
        logger.error(f"Demo data generation error: {str(e)}")
        return jsonify({'error': 'Error generating demo data'}), 500

@app.route('/api/drugs', methods=['GET'])
def get_drugs_list():
    """Get list of available drugs for forecasting"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        drugs = processed_data['Drug'].unique().tolist()
        return jsonify({'drugs': sorted(drugs)})
    
    except Exception as e:
        logger.error(f"Error getting drugs list: {str(e)}")
        return jsonify({'error': 'Error retrieving drugs list'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_drug():
    """
    New consolidated endpoint for frontend.
    Analyzes a specific drug, generates insights, and runs forecasts.
    """
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data available'}), 400
        
    try:
        request_data = request.get_json()
        drug = request_data.get('drug')
        
        if not drug:
            return jsonify({'error': 'Drug name is required'}), 400

        drug_data = processed_data[processed_data['Drug'] == drug].copy()
        if drug_data.empty:
            return jsonify({'error': f'No data found for drug: {drug}'}), 400

        trends_data = analytics_engine.get_trends_data(drug_data, 'monthly')
        historical_data = trends_data['trends'][0]['data'] if trends_data['trends'] else []
        for item in historical_data:
            item['date'] = item.pop('period')
            item['sales'] = item.pop('Revenue')

        insights = insights_generator.generate_insights(drug_data, max_insights=5)
        metrics = analytics_engine.calculate_metrics(drug_data)

        # Generate forecasts - this will now include REAL accuracy
        arima_forecast = forecasting_engine.generate_forecast(drug_data, model_name='arima', periods=12)
        prophet_forecast = forecasting_engine.generate_forecast(drug_data, model_name='prophet', periods=18)
        
        # Get real accuracy, defaulting to 0 if calculation fails
        arima_accuracy_perc = arima_forecast.get('accuracy', {}).get('accuracy_percentage', 0)
        prophet_accuracy_perc = prophet_forecast.get('accuracy', {}).get('accuracy_percentage', 0)

        response = {
            'analysisData': {
                'historical': historical_data,
                'trends': {
                    'overall_growth': f"{metrics.get('growth_rate', 0):.1f}%",
                    'seasonal_pattern': next((i['description'] for i in insights if i['type'] == 'pattern'), 'Seasonality not prominent.'),
                    'yearly_growth': [] 
                },
                'insights': insights
            },
            'forecastData': {
                'arima': arima_forecast.get('forecast'),
                'prophet': prophet_forecast.get('forecast'),
                'comparison': {
                    'arima_accuracy': f"{arima_accuracy_perc:.1f}%",
                    'prophet_accuracy': f"{prophet_accuracy_perc:.1f}%",
                    'recommended': 'ARIMA for short-term, Prophet for long-term strategic planning.'
                }
            }
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Error during analysis: {str(e)}'}), 500

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )