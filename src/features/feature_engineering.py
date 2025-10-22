### features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import pickle
from typing import Dict, Tuple, List
import logging
import os
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineer:
    """Handles feature extraction from text data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.svd = None
        
    def extract_basic_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract basic text features"""
        feature_df = df.copy()
        
        # Text length features
        feature_df['text_length'] = feature_df[text_column].str.len()
        feature_df['word_count'] = feature_df[text_column].str.split().str.len()
        feature_df['sentence_count'] = feature_df[text_column].str.count(r'\.')
        
        # Average word length
        feature_df['avg_word_length'] = (
            feature_df[text_column].str.replace(' ', '').str.len() / 
            feature_df['word_count']
        )
        
        # Punctuation features
        feature_df['exclamation_count'] = feature_df[text_column].str.count('!')
        feature_df['question_count'] = feature_df[text_column].str.count(r'\?')
        feature_df['uppercase_count'] = feature_df[text_column].str.count(r'[A-Z]')
        
        # Sentiment polarity using TextBlob
        feature_df['polarity'] = feature_df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        feature_df['subjectivity'] = feature_df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        
        self.logger.info(f"Extracted {len(feature_df.columns) - len(df.columns)} basic features")
        return feature_df
        
    def extract_tfidf_features(self, texts: List[str], max_features: int = 500):
        """Extract TF-IDF features"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            
        self.logger.info(f"Extracted TF-IDF features: {tfidf_features.shape}")
        return tfidf_features
        
    def extract_sentence_embeddings(self, texts: List[str], 
                                  model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Extract sentence embeddings using pre-trained models"""
        if self.sentence_transformer is None:
            self.sentence_transformer = SentenceTransformer(model_name)
            
        embeddings = self.sentence_transformer.encode(texts, batch_size=16, show_progress_bar=True)
        self.logger.info(f"Extracted sentence embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)
        
    def reduce_dimensionality(self, features: np.ndarray, 
                            n_components: int = 50) -> np.ndarray:
        """Reduce feature dimensionality using SVD"""
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_features = self.svd.fit_transform(features)
        else:
            reduced_features = self.svd.transform(features)
            
        self.logger.info(f"Reduced dimensions from {features.shape[1]} to {n_components}")
        return reduced_features.astype(np.float32)
        
    def combine_features(self, basic_features: pd.DataFrame, 
                        tfidf_features,
                        embeddings: np.ndarray = None):
        """Combine features in a memory-safe way (no dense conversion)"""
        numeric_cols = basic_features.select_dtypes(include=[np.number]).columns.tolist()
        basic_sparse = csr_matrix(basic_features[numeric_cols].fillna(0).values.astype(np.float32))

        if embeddings is not None:
            embeddings_sparse = csr_matrix(embeddings)
            combined = hstack([basic_sparse, tfidf_features, embeddings_sparse]).tocsr()
            self.logger.info(f"Combined sparse features (with embeddings): {combined.shape}")
        else:
            combined = hstack([basic_sparse, tfidf_features]).tocsr()
            self.logger.info(f"Combined sparse features: {combined.shape}")

        return combined

    def engineer_features(self, df: pd.DataFrame, text_column: str,
                         include_embeddings: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Complete feature engineering pipeline"""
        self.logger.info("Starting feature engineering...")
        
        # Extract basic features
        feature_df = self.extract_basic_features(df, text_column)
        
        # Cache TF-IDF features
        tfidf_path = "data/processed/features_tfidf.pkl"
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                tfidf_features = pickle.load(f)
            self.logger.info("Loaded cached TF-IDF features.")
        else:
            tfidf_features = self.extract_tfidf_features(df[text_column].tolist())
            with open(tfidf_path, "wb") as f:
                pickle.dump(tfidf_features, f)
        
        embeddings = None
        if include_embeddings:
            embeddings = self.extract_sentence_embeddings(df[text_column].tolist())
            embeddings = self.reduce_dimensionality(embeddings, n_components=50)
        
        X = self.combine_features(feature_df, tfidf_features, embeddings)
        
        # Labels
        y = None
        if 'sentiment' in df.columns:
            label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            y = df['sentiment'].map(label_mapping).values
        
        self.logger.info("Feature engineering complete")
        return X, y
        
    def save_feature_extractors(self, filepath: str):
        """Save feature extractors for later use"""
        extractors = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'sentence_transformer': self.sentence_transformer,
            'svd': self.svd
        }
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        self.logger.info(f"Feature extractors saved to {filepath}")
        
    def load_feature_extractors(self, filepath: str):
        """Load pre-trained feature extractors"""
        with open(filepath, 'rb') as f:
            extractors = pickle.load(f)
        self.tfidf_vectorizer = extractors['tfidf_vectorizer']
        self.sentence_transformer = extractors['sentence_transformer']
        self.svd = extractors['svd']
        self.logger.info(f"Feature extractors loaded from {filepath}")
