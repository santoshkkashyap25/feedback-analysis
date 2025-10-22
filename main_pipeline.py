
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add src to Python path
sys.path.append('src')

from data.ingestion import DataIngestionPipeline
from data.preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.training import ModelTrainer
from models.evaluation import ModelEvaluator
from monitoring.drift_detection import ModelMonitor
from monitoring.drift_detection import RetrainingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FeedbackAnalysisPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run_data_ingestion(self):
        """Run data ingestion step"""
        self.logger.info("Starting data ingestion...")
        
        ingestion = DataIngestionPipeline(self.config)
        
        # Define data sources
        sources = [
            {
                'type': 'kaggle',
                'name': 'amazon_reviews',
                'dataset': 'arhamrumi/amazon-product-reviews',
                'download_path': 'data/raw',
                'filename': 'Reviews.csv'
            }
        ]

        # sources = [
        #     {
        #         "type": "csv",
        #         "name": "local_reviews",
        #         "path": "data/raw/sample_reviews.csv"
        #     }
        # ]

        
        # Ingest data
        raw_data = ingestion.ingest_multiple_sources(sources)
        
        # Save raw data
        raw_data.to_csv('data/raw/combined_reviews.csv', index=False)
        self.logger.info(f"Raw data saved: {len(raw_data)} records")
        
        return raw_data
        
    def run_preprocessing(self, raw_data):
        """Run data preprocessing step"""
        self.logger.info("Starting data preprocessing...")
        
        preprocessor = DataPreprocessor(self.config)
        processed_data = preprocessor.preprocess_pipeline(raw_data)
        
        # Save processed data
        processed_data.to_csv('data/processed/processed_reviews.csv', index=False)
        self.logger.info(f"Processed data saved: {len(processed_data)} records")
        
        return processed_data
        
    def run_feature_engineering(self, processed_data):
        """Run feature engineering step"""
        self.logger.info("Starting feature engineering...")
        
        feature_engineer = FeatureEngineer(self.config)
        
        # Extract features
        X, y = feature_engineer.engineer_features(
            processed_data, 
            'reviewText_clean',
            include_embeddings=False
        )
        
        # Save feature extractors
        feature_engineer.save_feature_extractors('data/models/feature_extractors.pkl')
        
        return X, y, feature_engineer
        
    def run_model_training(self, X, y):
        """Run model training step"""
        self.logger.info("Starting model training...")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        # Handle class imbalance
        preprocessor = DataPreprocessor(self.config)
        X_train_balanced, y_train_balanced = preprocessor.handle_imbalanced_data(
            X_train, y_train, strategy='smote'
        )
        
        # Train models
        trainer = ModelTrainer(self.config)
        trained_models = trainer.train_all_models(X_train_balanced, y_train_balanced)
        
        # Select best model
        best_model_name, best_model = trainer.select_best_model(X_val, y_val)
        
        # Save best model
        trainer.save_model(best_model, 'data/models/best_model.pkl')
        
        return best_model, (X_train, X_val, X_test, y_train, y_val, y_test), trained_models
        
    def run_model_evaluation(self, model, data_splits, trained_models):
        """Run model evaluation step"""
        self.logger.info("Starting model evaluation...")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        
        evaluator = ModelEvaluator()
        
        # Comprehensive evaluation on test set
        test_results = evaluator.evaluate_model_comprehensive(
            model, X_test, y_test
        )
        
        # Compare all models
        model_comparison = evaluator.compare_models(trained_models, X_test, y_test)
        
        self.logger.info("Model Comparison Results:")
        self.logger.info(f"\n{model_comparison}")
        
        return test_results, model_comparison
        
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            # Create directories
            os.makedirs('data/raw', exist_ok=True)
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('data/models', exist_ok=True)
            os.makedirs('logs', exist_ok=True)

            # Define checkpoint paths
            raw_data_path = "data/raw/Reviews.csv"
            processed_data_path = "data/processed/processed_reviews.csv"

            
            # Data Ingestion
            if os.path.exists(raw_data_path):
                self.logger.info(f"Found existing raw dataset at {raw_data_path}, skipping ingestion.")
                raw_data = pd.read_csv(raw_data_path)
            else:
                raw_data = self.run_data_ingestion()
            
            # Preprocessing
            if os.path.exists(processed_data_path):
                self.logger.info(f"Found processed dataset at {processed_data_path}, skipping preprocessing.")
                processed_data = pd.read_csv(processed_data_path)
            else:
                processed_data = self.run_preprocessing(raw_data)

            MAX_ROWS = 5000
            if len(processed_data) > MAX_ROWS:
                self.logger.info(f"Limiting dataset from {len(processed_data)} to {MAX_ROWS} rows for memory efficiency.")
                processed_data = processed_data.groupby('sentiment', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), int(MAX_ROWS/3)), random_state=self.config.get('random_state', 42))
                ).reset_index(drop=True)

            
            # Feature Engineering
            X, y, feature_engineer = self.run_feature_engineering(processed_data)
            
            # Model Training
            best_model, data_splits, trained_models = self.run_model_training(X, y)
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
            
            # Model Evaluation
            test_results, model_comparison = self.run_model_evaluation(
                best_model, data_splits, trained_models
            )
            
            # --- Monitoring ---
            monitor = ModelMonitor()
            baseline_accuracy = test_results['basic_metrics']['accuracy']
            baseline_f1 = test_results['basic_metrics']['f1']

            monitor.log_performance_metrics(
                accuracy=baseline_accuracy,
                f1=baseline_f1,
                sample_size=len(y_test)
            )

            self.logger.info(f"Logged performance: acc={baseline_accuracy:.4f}, f1={baseline_f1:.4f}")

            # --- Drift Detection ---
            ref_path = "data/processed/features_tfidf.pkl"
            if os.path.exists(ref_path):
                import pickle
                with open(ref_path, "rb") as f:
                    reference_features = pickle.load(f)
                drift_result = monitor.detect_data_drift(reference_features, X)
                self.logger.info(f"Drift detected? {drift_result['overall_drift']}")
            else:
                import pickle
                with open(ref_path, "wb") as f:
                    pickle.dump(X, f)
                self.logger.info("Saved reference features for future drift comparison.")

            # --- Retraining Decision ---
            from monitoring.drift_detection import RetrainingStrategy
            strategy = RetrainingStrategy(config=self.config)
            retrain_decision = strategy.should_retrain(monitor)

            if retrain_decision["should_retrain"]:
                self.logger.warning(f"Retraining triggered: {retrain_decision['reasons']}")

                # === AUTOMATIC RETRAINING ===
                retraining_plan = strategy.create_retraining_plan(
                    new_data_size=len(processed_data),
                    current_model_performance={'accuracy': baseline_accuracy, 'f1': baseline_f1}
                )

                self.logger.info("Starting retraining process...")
                best_model, data_splits, trained_models = self.run_model_training(X, y)
                X_train, X_val, X_test, y_train, y_val, y_test = data_splits
                test_results, model_comparison = self.run_model_evaluation(best_model, data_splits, trained_models)

                # Save new model
                model_path = "data/models/best_model_retrained.pkl"
                import joblib
                joblib.dump(best_model, model_path)
                self.logger.info(f"Retrained model saved at {model_path}")

                # Update reference features
                with open(ref_path, "wb") as f:
                    pickle.dump(X, f)

                monitor.log_performance_metrics(
                    accuracy=test_results['basic_metrics']['accuracy'],
                    f1=test_results['basic_metrics']['f1'],
                    sample_size=len(y_test)
                )

                self.logger.info("Retraining completed successfully.")

            else:
                self.logger.info("Model healthy — no retraining required.")

            self.logger.info("Pipeline completed successfully!")

            return {
                'model': best_model,
                'feature_engineer': feature_engineer,
                'test_results': test_results,
                'model_comparison': model_comparison,
                'monitor': monitor,
                'retrain_decision': retrain_decision
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Customer Feedback Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip data download if files exist')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'validation_size': 0.2,
        'max_features': 5000,
        'embedding_model': 'all-MiniLM-L6-v2',
        'resampling_strategy': 'smote'
    }
    
    # Load config file if provided
    if os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Run pipeline
    pipeline = FeedbackAnalysisPipeline(config)
    results = pipeline.run_full_pipeline()
    
    print(f"Best Model Performance:")
    print(f"Accuracy: {results['test_results']['basic_metrics']['accuracy']:.4f}")
    print(f"F1-Score: {results['test_results']['basic_metrics']['f1']:.4f}")
    print(f"Precision: {results['test_results']['basic_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_results']['basic_metrics']['recall']:.4f}")

    retrain_decision = results['retrain_decision']
    print("\n=== MONITORING SUMMARY ===")
    print(f"Should Retrain? {retrain_decision['should_retrain']}")
    if retrain_decision['reasons']:
        print(f"Reasons: {', '.join(retrain_decision['reasons'])}")
    
if __name__ == '__main__':
    main()