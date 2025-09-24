import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
def preprocess_data(df: pd.DataFrame, target_column: str) -> (pd.DataFrame, pd.Series):
    """
    Preprocess the data by handling missing values and separating features and target.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
    Returns:
        pd.DataFrame: Features DataFrame.
        pd.Series: Target Series.
    """
    try:
        df = df.dropna()
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
        
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise  
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: str, test_path: str):
    """
    Save the training and testing DataFrames to CSV files.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        train_path (str): Path to save the training CSV file.
        test_path (str): Path to save the testing CSV file.
    """
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Training data saved to {train_path}")
        logger.info(f"Testing data saved to {test_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
def main():
    try:
        # Load data
        data = load_data(r'C:\Users\lenovo\Desktop\Mlops-complte-on-Medical-Insurance-Cost-Dataset\data\insurance.csv')
        
        # Preprocess data
        data = preprocess_data(data, target_column='charges')
        
        # Split data into training and testing sets
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
        
        # Save processed data
        save_data(train_df, test_df, 'data/processed/train.csv', 'data/processed/test.csv')
        
        logger.info("Data ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Data ingestion process failed: {e}")


        

if __name__ == "__main__":
    main()
