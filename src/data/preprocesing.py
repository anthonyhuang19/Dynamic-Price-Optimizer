import pandas as pd
from load import *
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class responsible for data preprocessing tasks."""
    
    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame

    def split_numerical_object(self):
        """Splits the dataframe into numerical and categorical columns."""
        obj = []
        num = []

        # Iterate through columns to classify based on dtype
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                obj.append((col, self.df[col].nunique(), self.df[col].isna().sum()))  # Unique values and missing values
            else:
                num.append((col, self.df[col].nunique(), self.df[col].isna().sum(), self.df[col].skew()))  # Skewness

        # Make lists the same length by extending with empty tuples
        max_length = max(len(obj), len(num))
        obj.extend([('', '', '')] * (max_length - len(obj)))
        num.extend([('', '', '', '')] * (max_length - len(num)))

        # Prepare the result into a DataFrame
        data = {
            'Category Columns': [x[0] for x in obj],
            'Category Unique': [x[1] for x in obj],
            'Category Sum Missing Val': [x[2] for x in obj],
            'Numerical Columns': [x[0] for x in num],
            'Numerical Unique': [x[1] for x in num],
            'Numerical Sum Missing Val': [x[2] for x in num],
            'Numerical Skew': [x[3] for x in num]
        }
        
        return pd.DataFrame(data)

    def show_data_info(self):
        """Displays information about the dataframe."""
        logger.info("Displaying DataFrame Info:")
        print(self.df.info())

    def show_duplicates(self):
        """Displays the count of duplicate rows in the DataFrame."""
        duplicates_count = self.df.duplicated().value_counts()
        logger.info(f"Duplicate row count:\n{duplicates_count}")
        print(f"Duplicate row count:\n{duplicates_count}")

    def preprocess(self):
        """Runs the full preprocessing pipeline."""
        # Display info about the data
        self.show_data_info()

        # Display duplicate values in the DataFrame
        self.show_duplicates()

        # Get the split dataframe with numerical and categorical columns
        df_split = self.split_numerical_object()
        logger.info("Displaying split numerical and categorical columns information:")
        print(df_split)

        
def main():
    """Main function to run the preprocessing."""
    data_path = "data/raw/dynamic_pricing.csv"
    
    # Load the CSV data
    data_loader = DataLoader(data_path)
    df = data_loader.load()

    if df is not None:
        # Create a DataPreprocessor instance and run the preprocessing
        preprocessor = DataPreprocessor(df)
        preprocessor.preprocess()
    else:
        logger.error("Data loading failed. Exiting the program.")
        
if __name__ == "__main__":
    main()
