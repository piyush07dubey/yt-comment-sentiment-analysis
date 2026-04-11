import pandas as pd
import yaml
import os
import logging
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, test_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        df = pd.read_csv(test_path)
        X_test = df.drop('sentiment', axis=1)
        y_test = df['sentiment']
        
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error in evaluating model: {e}")
        raise e

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {path}")

def main():
    params_path = "params.yaml"
    params = load_params(params_path)
    
    eval_params = params['model_evaluation']
    train_params = params['model_training']
    fe_params = params['feature_engineering']
    
    metrics = evaluate_model(
        train_params['model_path'],
        fe_params['test_features_path']
    )
    
    save_metrics(metrics, eval_params['metrics_path'])

if __name__ == "__main__":
    main()
