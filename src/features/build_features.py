import pandas as pd
import yaml
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def build_features(train_path, test_path, max_features):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info("Building features using TF-IDF...")
        tfidf = TfidfVectorizer(max_features=max_features)
        
        # We assume 'clean_comment' exists from the previous stage
        X_train = tfidf.fit_transform(train_df['clean_comment']).toarray()
        X_test = tfidf.transform(test_df['clean_comment']).toarray()
        
        train_features_df = pd.DataFrame(X_train)
        train_features_df['sentiment'] = train_df['sentiment'].values
        
        test_features_df = pd.DataFrame(X_test)
        test_features_df['sentiment'] = test_df['sentiment'].values
        
        return train_features_df, test_features_df
    except Exception as e:
        logger.error(f"Error in building features: {e}")
        raise e

def save_features(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Features saved to {path}")

def main():
    params_path = "params.yaml"
    params = load_params(params_path)
    
    fe_params = params['feature_engineering']
    di_params = params['data_ingestion']
    
    train_features_df, test_features_df = build_features(
        di_params['train_data_path'],
        di_params['test_data_path'],
        fe_params['max_features']
    )
    
    save_features(train_features_df, fe_params['train_features_path'])
    save_features(test_features_df, fe_params['test_features_path'])

if __name__ == "__main__":
    main()
