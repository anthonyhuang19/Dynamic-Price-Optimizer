import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from load import *
import joblib


class FeatureEngineering:
    """Class to handle feature engineering processes."""
    
    @staticmethod
    def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering on the DataFrame and return modified df."""
        df['demand_multiplier'] = FeatureEngineering._calculate_demand_multiplier(df)
        df['supply_multiplier'] = FeatureEngineering._calculate_supply_multiplier(df)
        df['adjusted_ride_cost'] = FeatureEngineering._calculate_adjusted_ride_cost(df)
        df['profit_percentage'] = FeatureEngineering._calculate_profit_percentage(df)
        return df

    @staticmethod
    def _calculate_demand_multiplier(df: pd.DataFrame) -> pd.Series:
        high_demand = 75
        low_demand = 25
        return np.where(
            df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], high_demand),
            df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand),
            df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand)
        )

    @staticmethod
    def _calculate_supply_multiplier(df: pd.DataFrame) -> pd.Series:
        low_demand = 25
        high_demand = 75
        return np.where(
            df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_demand),
            np.percentile(df['Number_of_Drivers'], high_demand) / df['Number_of_Drivers'],
            np.percentile(df['Number_of_Drivers'], low_demand) / df['Number_of_Drivers']
        )

    @staticmethod
    def _calculate_adjusted_ride_cost(df: pd.DataFrame) -> pd.Series:
        demand_threshold_low = 0.8
        supply_threshold_high = 0.8
        return df['Historical_Cost_of_Ride'] * (
            np.maximum(df['demand_multiplier'], demand_threshold_low) *
            np.maximum(df['supply_multiplier'], supply_threshold_high)
        )

    @staticmethod
    def _calculate_profit_percentage(df: pd.DataFrame) -> pd.Series:
        return ((df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride']) / df['Historical_Cost_of_Ride']) * 100


class DataTransformation:
    """Class for handling data transformation and mappings."""
    
    @staticmethod
    def transform_features(df: pd.DataFrame) -> pd.DataFrame:
        """Perform data transformation and mapping on the DataFrame."""
        df['Location_Category'] = DataTransformation._map_location_category(df['Location_Category'])
        df['Customer_Loyalty_Status'] = DataTransformation._map_customer_loyalty(df['Customer_Loyalty_Status'])
        df['Vehicle_Type'] = DataTransformation._map_vehicle_type(df['Vehicle_Type'])
        df['Time_of_Booking'] = DataTransformation._map_time_of_booking(df['Time_of_Booking'])
        df['interpolated_division'] = DataTransformation._apply_polynomial_interpolation(df)
        return df

    @staticmethod
    def _map_location_category(series: pd.Series) -> pd.Series:
        Location_Mapping = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
        return series.map(Location_Mapping)

    @staticmethod
    def _map_customer_loyalty(series: pd.Series) -> pd.Series:
        Customer_Loyalty_Status_Mapping = {'Silver': 0, 'Regular': 1, 'Gold': 2}
        return series.map(Customer_Loyalty_Status_Mapping)

    @staticmethod
    def _map_vehicle_type(series: pd.Series) -> pd.Series:
        Vehicle_Mapping = {'Premium': 0, 'Economy': 1}
        return series.map(Vehicle_Mapping)

    @staticmethod
    def _map_time_of_booking(series: pd.Series) -> pd.Series:
        time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
        return series.map(time_mapping)

    @staticmethod
    def _apply_polynomial_interpolation(df: pd.DataFrame) -> pd.Series:
        coefficients = np.polyfit(df['Number_of_Riders'].values, df['Number_of_Drivers'].values, deg=2)
        poly = np.poly1d(coefficients)
        joblib.dump(poly, 'model/polynomial_model.pkl')
        return poly(df['Number_of_Riders'].values / df['Number_of_Drivers'].values)


class DataVisualization:
    """Class for handling data visualization."""
    
    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, save_dir: str = "images"):
        """Plot histogram and KDE for a column with mean and median lines, save as image."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(8, 6))
        mean_values = df[column].mean()
        median = df[column].median()

        plt.axvline(x=mean_values, color='#F28585', linestyle='--', label='Mean')
        plt.axvline(x=median, color='#747264', linestyle='--', label='Median')
        sns.histplot(df[column], kde=True, color='#638889')
        plt.grid(True)
        plt.legend()
        plt.title(f'Distribution of {column}')

        filename = f"{column}_distribution.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()  # Close figure to free memory
        print(f"Saved plot to {filepath}")


class DataProcessing:
    """Main class for processing and saving data."""
    
    @staticmethod
    def process_data(raw_csv_path: str, processed_csv_path: str):
        data_loader = DataLoader(raw_csv_path)
        df = data_loader.load()

        # Feature Engineering
        df = FeatureEngineering.apply_feature_engineering(df)

        # Profitability analysis printout
        DataProcessing._print_profitability_analysis(df)

        # Before log transform
        DataProcessing._print_skewness(df)
        DataVisualization.plot_distribution(df, 'adjusted_ride_cost', save_dir="src/data/images")

        # Apply log transform
        df = DataProcessing._apply_log_transform(df)

        # After log transform
        DataProcessing._print_skewness(df)
        DataVisualization.plot_distribution(df, 'adjusted_ride_cost', save_dir="src/data/images")

        # Data transformation mappings and new features
        df = DataTransformation.transform_features(df)

        # Save processed data
        df.to_csv(processed_csv_path, index=False)
        print(f"Processed data saved to {processed_csv_path}")

    @staticmethod
    def _print_profitability_analysis(df: pd.DataFrame):
        profitable_rides = df[df['profit_percentage'] > 0]
        loss_ride = df[df['profit_percentage'] < 0]
        count = len(profitable_rides) + len(loss_ride)
        print(f"Positive dynamic pricing ratio: {len(profitable_rides)/count:.2f}")
        print(f"Negative profit ratio: {len(loss_ride)/count:.2f}")

    @staticmethod
    def _print_skewness(df: pd.DataFrame):
        print(f"Skewness before log transform: {df['adjusted_ride_cost'].skew():.2f}")

    @staticmethod
    def _apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
        df['adjusted_ride_cost'] = np.log1p(df['adjusted_ride_cost'])
        return df


if __name__ == "__main__":
    RAW_CSV = "data/raw/dynamic_pricing.csv"
    PROCESSED_CSV = "data/processed/data.csv"
    DataProcessing.process_data(RAW_CSV, PROCESSED_CSV)
