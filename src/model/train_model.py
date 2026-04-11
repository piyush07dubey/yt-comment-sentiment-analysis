import pandas as pd
import yaml
import os
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(train_path, n_estimators, max_depth):
    try:
        df = pd.read_csv(train_path)
        X_train = df.drop('sentiment', axis=1)
        y_train = df['sentiment']
        
        logger.info(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}...")
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        return clf
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise e

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")

def main():
    params_path = "params.yaml"
    params = load_params(params_path)
    
    train_params = params['model_training']
    fe_params = params['feature_engineering']
    
    model = train_model(
        fe_params['train_features_path'],
        train_params['n_estimators'],
        train_params['max_depth']
    )
    
    save_model(model, train_params['model_path'])

if __name__ == "__main__":
    main()
