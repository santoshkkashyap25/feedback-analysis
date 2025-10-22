### tests/test_preprocessing.py

import unittest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings("ignore")

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor({})
        self.sample_data = pd.DataFrame({
            'reviewText': [
                'This product is amazing! I love it.',
                'Terrible quality. Very disappointed.',
                'This product is amazing! I love it.',  # Duplicate
                '',  # Empty
                None,  # Null
                'Average product, nothing special.'
            ],
            'overall': [5, 1, 5, 3, None, 3]
        })
        
    def test_remove_duplicates(self):
        result = self.preprocessor.remove_duplicates(self.sample_data)
        self.assertEqual(len(result), 5)  # Should remove 1 duplicate
        
    def test_handle_missing_values(self):
        result = self.preprocessor.handle_missing_values(self.sample_data)
        # Should remove rows with missing reviewText
        self.assertFalse(result['reviewText'].isna().any())
        
    def test_clean_text(self):
        test_text = "This is AMAZING!!! <br> Very good product... 123"
        result = self.preprocessor.clean_text(test_text)
        
        # Should be lowercase, no HTML, no numbers
        self.assertTrue(result.islower())
        self.assertNotIn('<br>', result)
        self.assertNotIn('123', result)
        
    def test_create_sentiment_labels(self):
        result = self.preprocessor.create_sentiment_labels(
            self.sample_data, 'overall'
        )
        
        # Check sentiment mapping
        self.assertEqual(result.loc[0, 'sentiment'], 'Positive')  # Rating 5
        self.assertEqual(result.loc[1, 'sentiment'], 'Negative')  # Rating 1
        
if __name__ == '__main__':
    unittest.main()
