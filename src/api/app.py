
### api/app.py

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Any
import os
from datetime import datetime
import time
from functools import wraps
import redis
from concurrent.futures import ThreadPoolExecutor
import threading


import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and feature extractors
model = None
feature_engineer = None
redis_client = None
request_count = 0
request_lock = threading.Lock()

class FeedbackAnalysisAPI:
    """Main API class for feedback analysis"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.load_model_and_extractors()
        
    def load_model_and_extractors(self):
        """Load pre-trained model and feature extractors"""
        try:
            # Load model
            model_path = os.getenv('MODEL_PATH', 'data/models/best_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            
            # Load feature extractors
            extractors_path = os.getenv('EXTRACTORS_PATH', 'data/models/feature_extractors.pkl')
            with open(extractors_path, 'rb') as f:
                extractors = pickle.load(f)
            
            # Initialize feature engineer with loaded extractors
            from src.features.feature_engineering import FeatureEngineer
            self.feature_engineer = FeatureEngineer({})
            self.feature_engineer.tfidf_vectorizer = extractors['tfidf_vectorizer']
            self.feature_engineer.sentence_transformer = extractors['sentence_transformer']
            self.feature_engineer.svd = extractors['svd']
            
            logger.info("Feature extractors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model/extractors: {e}")
            raise
            
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        from src.data.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor({})
        return preprocessor.clean_text(text)
        
    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from preprocessed text"""
        # Create temporary dataframe
        df = pd.DataFrame({'reviewText_clean': [text]})
        
        # Extract basic features
        feature_df = self.feature_engineer.extract_basic_features(df, 'reviewText_clean')
        
        # Extract TF-IDF features
        tfidf_features = self.feature_engineer.extract_tfidf_features([text])
        
        # Extract embeddings
        embeddings = self.feature_engineer.extract_sentence_embeddings([text])
        embeddings = self.feature_engineer.reduce_dimensionality(embeddings, n_components=50)
        
        # Combine features
        features = self.feature_engineer.combine_features(feature_df, tfidf_features, embeddings)
        
        return features
        
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given text"""
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            if not clean_text.strip():
                return {
                    'sentiment': 'Neutral',
                    'confidence': 0.33,
                    'probabilities': {'Negative': 0.33, 'Neutral': 0.34, 'Positive': 0.33},
                    'error': 'Empty text after preprocessing'
                }
            
            # Extract features
            features = self.extract_features(clean_text)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Map prediction to sentiment
            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = sentiment_map[prediction]
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Format probabilities
            prob_dict = {
                'Negative': float(probabilities[0]),
                'Neutral': float(probabilities[1]),
                'Positive': float(probabilities[2])
            }
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': prob_dict,
                'processed_text': clean_text
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'sentiment': 'Unknown',
                'confidence': 0.0
            }

# Initialize API
api = FeedbackAnalysisAPI()

# Rate limiting decorator
def rate_limit(max_requests=100, window=3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            global request_count
            
            with request_lock:
                request_count += 1
                
            if request_count > max_requests:
                return jsonify({'error': 'Rate limit exceeded'}), 429
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Monitoring decorator
def monitor_requests(f):
    """Request monitoring decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            status = 'success'
            response_time = time.time() - start_time
            
            # Log metrics
            logger.info(f"Request completed - Status: {status}, Response time: {response_time:.3f}s")
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Request failed - Error: {e}, Response time: {response_time:.3f}s")
            raise
            
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': api.model is not None
    })

@app.route('/predict', methods=['POST'])
@rate_limit(max_requests=1000, window=3600)
@monitor_requests
def predict():
    """Single prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        # Make prediction
        result = api.predict_sentiment(text)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['model_version'] = '1.0'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/batch', methods=['POST'])
@rate_limit(max_requests=100, window=3600)
@monitor_requests
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size cannot exceed 100'}), 400
        
        # Process predictions in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(api.predict_sentiment, texts))
        
        # Add metadata
        response = {
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    return jsonify({
        'total_requests': request_count,
        'uptime': datetime.now().isoformat(),
        'model_info': {
            'version': '1.0',
            'type': 'sentiment_classifier',
            'classes': ['Negative', 'Neutral', 'Positive']
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)