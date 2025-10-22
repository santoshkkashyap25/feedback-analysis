### data/preprocessing.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Dict
import logging
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Original Columns: ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 
# 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text', 'data_source']

class DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=['reviewText'], keep='first')
        removed_count = initial_count - len(df_clean)
        self.logger.info(f"Removed {removed_count} duplicate records")
        return df_clean
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Remove rows where review text is missing
        df_clean = df.dropna(subset=['reviewText'])
        
        # Fill missing ratings with median
        if 'overall' in df.columns:
            df_clean['overall'].fillna(df_clean['overall'].median(), inplace=True)
            
        # Fill missing summaries with empty string
        if 'summary' in df.columns:
            df_clean['summary'].fillna('', inplace=True)
            
        self.logger.info(f"Handled missing values, {len(df_clean)} records remaining")
        return df_clean
        
    def clean_text(self, text: str) -> str:
        """Clean individual text review"""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
        
    def normalize_text(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Normalize text data"""
        self.logger.info("Starting text normalization...")
        df[f'{text_column}_clean'] = df[text_column].apply(self.clean_text)
        
        # Remove empty reviews after cleaning
        df = df[df[f'{text_column}_clean'].str.len() > 0]
        
        self.logger.info(f"Text normalization complete, {len(df)} records remaining")
        return df
        
    def create_sentiment_labels(self, df: pd.DataFrame, rating_column: str) -> pd.DataFrame:
        """Create sentiment labels from ratings"""
        def rating_to_sentiment(rating):
            if rating <= 2:
                return 'Negative'
            elif rating >= 4:
                return 'Positive'
            else:
                return 'Neutral'
                
        df['sentiment'] = df[rating_column].apply(rating_to_sentiment)
        return df
        
    def handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray, 
                              strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle imbalanced dataset using various techniques"""
        
        self.logger.info(f"Original class distribution: {np.bincount(y)}")
        
        if strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        elif strategy == 'combined':
            # First oversample minority classes, then undersample majority
            smote = SMOTE(random_state=42)
            X_temp, y_temp = smote.fit_resample(X, y)
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
        else:
            X_resampled, y_resampled = X, y
            
        self.logger.info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
        
    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline...")

        # Standardize column names
        df = df.rename(columns={
            'Text': 'reviewText',
            'Score': 'overall',
            'Summary': 'summary'
        })
        
        # convert time
        if 'Time' in df.columns:
            df['review_time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')        
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Normalize text
        df = self.normalize_text(df, 'reviewText')
        
        # Create sentiment labels
        if 'overall' in df.columns:
            df = self.create_sentiment_labels(df, 'overall')
        
        self.logger.info("Preprocessing pipeline complete")
        return df