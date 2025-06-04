import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from load import *

# Create a directory to save images if it doesn't exist
save_dir = 'src/data/images'
os.makedirs(save_dir, exist_ok=True)

def plot_histogram_distribution(df: pd.DataFrame):
    """Plots histograms for numerical columns and saves the plot as an image."""
    numerical = df.select_dtypes(include='number')
    num_plots = len(numerical.columns)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

    plt.figure(figsize=(10, 4 * num_rows))

    for i, col in enumerate(numerical, 1):
        plt.subplot(num_rows, num_columns, i)
        mean_values = numerical[col].mean()
        median = numerical[col].median()

        sns.histplot(numerical[col], kde=True, color='#638889')
        plt.axvline(x=mean_values, color='#F28585', linestyle='--', label='Mean')
        plt.axvline(x=median, color='#747264', linestyle='--', label='Median')
        plt.grid(True, alpha=0.8)
        plt.title(f'{col} Distribution')
        plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'numerical_histograms.png'))
   


def plot_categorical_distribution(df: pd.DataFrame):
    """Plots histograms for categorical columns and saves the plot as an image."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    num_plots = len(categorical_cols)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

    plt.figure(figsize=(10, 4 * num_rows))

    for i, col in enumerate(df[categorical_cols], 1):
        mode = df[col].mode()[0]
        plt.subplot(num_rows, num_columns, i)
        sns.histplot(df[col], kde=True, color='#638889')
        plt.axvline(x=mode, color='#F28585', linestyle='--', label='Mode')
        plt.xticks(rotation=90, fontsize=7)
        plt.title(f'{col} Distribution')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'categorical_histograms.png'))
   


def plot_outliers(df: pd.DataFrame):
    """Plots Z-score distributions for numerical columns to detect outliers and saves the plot."""
    numerics = df.select_dtypes(include=np.number)
    num_plots = len(numerics.columns)
    num_columns = 3
    num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

    plt.figure(figsize=(10, 4 * num_rows))

    for i, col in enumerate(numerics, 1):
        plt.subplot(num_rows, num_columns, i)
        z_scores = (numerics[col] - numerics[col].mean()) / numerics[col].std()
        threshold = 3

        plt.scatter(np.arange(len(z_scores)), z_scores, color='#638889', alpha=0.5)
        plt.axhline(y=threshold, color='#F28585', linestyle='--', label='Threshold')
        plt.axhline(y=-threshold, color='#F28585', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Z-score')
        plt.title(f'Z-score Plot for {col}')
        plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'outlier_z_scores.png'))
   


def plot_correlation_matrix(df: pd.DataFrame):
    """Plots a heatmap of the correlation matrix for numerical columns and saves the plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', annot_kws={"fontsize": 8})
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    


def plot_regression_line(df: pd.DataFrame):
    """Plots a regression line between 'Expected_Ride_Duration' and 'Historical_Cost_of_Ride'."""
    sns.lmplot(data=df, y='Historical_Cost_of_Ride', x='Expected_Ride_Duration', hue='Vehicle_Type', palette=['#638889', '#f28585'])
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'regression_line.png'))
    


def main(data_path = "data/raw/dynamic_pricing.csv"):
    """Main function to load data, perform EDA, and save visualizations."""
    # Load data
    data_loader = DataLoader(data_path)
    df = data_loader.load()

    if df is not None:
        print(df.head())
        
        # Exploratory Data Analysis (EDA)
        plot_histogram_distribution(df)
        plot_categorical_distribution(df)
        plot_outliers(df)
        plot_correlation_matrix(df)
        plot_regression_line(df)


if __name__ == "__main__":
    main()
