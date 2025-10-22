### tests/test_api.py
import unittest
import json
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from api.app import app

class TestFeedbackAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        
    def test_predict_endpoint_valid_input(self):
        payload = {
            'text': 'This product is amazing! I love it so much.'
        }
        response = self.app.post('/predict', 
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)
        self.assertIn('probabilities', data)
        
    def test_predict_endpoint_invalid_input(self):
        # Test missing text field
        payload = {}
        response = self.app.post('/predict',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test empty text
        payload = {'text': ''}
        response = self.app.post('/predict',
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
    def test_batch_predict_endpoint(self):
        payload = {
            'texts': [
                'Great product!',
                'Terrible quality.',
                'Average item.'
            ]
        }
        response = self.app.post('/predict/batch',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['count'], 3)
        self.assertEqual(len(data['predictions']), 3)
        
if __name__ == '__main__':
    unittest.main()