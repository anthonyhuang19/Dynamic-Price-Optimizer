import os
import pandas as pd
import logging
from typing import Optional

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and validation."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> Optional[pd.DataFrame]:
        """Load CSV data from the given file path."""
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            return None

        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.exception(f"Failed to load data from {self.file_path}. Error: {str(e)}")
            return None


class DataProcessor:
    """Class to handle data processing operations."""
    
    @staticmethod
    def display_head(df: pd.DataFrame, num_rows: int = 5) -> None:
        """Display the top rows of the DataFrame."""
        if df is not None:
            print(f"First {num_rows} rows of the data:")
            print(df.head(num_rows))
        else:
            logger.warning("DataFrame is None, cannot display head.")
    

if __name__ == "__main__":
    # Define the path to the raw CSV file
    data_path = "data/raw/dynamic_pricing.csv"

    # Create a DataLoader instance and load the data
    data_loader = DataLoader(data_path)
    df = data_loader.load()

    if df is not None:
        DataProcessor.display_head(df)
    else:
        logger.error("Data loading failed. Exiting the program.")
