import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.metrics import make_scorer

# Data preparation
def data_splitting(path: str):
    df = pd.read_csv(path)
    Y = df['adjusted_ride_cost']
    Y = np.expm1(Y)
    X = df[['Number_of_Riders', 'Number_of_Drivers', 'interpolated_division', 'Location_Category', 
            'Vehicle_Type', 'Time_of_Booking', 'Expected_Ride_Duration']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'model/scaler.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Model initialization
def initializing_models():
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'XGB': XGBRegressor(objective='reg:squarederror'),
        'CatBoost': CatBoostRegressor(verbose=0),
        'SVR': SVR(),
        'MLP': MLPRegressor(max_iter=500)
    }
    pipelines = {
        name: Pipeline([
            ('scaler', StandardScaler()), 
            ('model', model)
        ]) for name, model in models.items()
    }
    params = {
        'LinearRegression': {},
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'model__alpha': [0.001, 0.01, 0.1]},
        'DecisionTree': {'model__max_depth': [3, 5, 10]},
        'RandomForest': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]},
        'XGB': {'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]},
        'CatBoost': {'model__depth': [4, 6], 'model__learning_rate': [0.1, 0.3]},
        'SVR': {'model__C': [0.1, 1], 'model__epsilon': [0.1, 0.2], 'model__kernel': ['linear', 'rbf']},
        'MLP': {'model__hidden_layer_sizes': [(64,), (128, 64)], 'model__activation': ['relu'], 'model__alpha': [0.001]}
    }
    return models, pipelines, params

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def train(pipelines, params, X_train, y_train, X_test, y_test):
    best_model = None
    best_rmse, best_r2 = math.inf, -math.inf
    best_model_name = None
    results = {}
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    for name, pipeline in pipelines.items():  # Fixed: iterate through pipelines.items()
        # Start a new MLflow run for each model
        with mlflow.start_run():
            if name in params:
                grid_search = GridSearchCV(pipeline, params[name], cv=5, scoring=rmse_scorer)
            else:
                grid_search = GridSearchCV(pipeline, {}, cv=5, scoring=rmse_scorer)
            
            grid_search.fit(X_train, y_train)  # Fixed: use grid_search instead of search
            model = grid_search.best_estimator_
            y_pred = model.predict(X_test)
            # Metrics - fixed indexing issue with y_test and y_pred
            current_rmse = rmse(y_test,y_pred)
            current_r2 = r2_score(y_test,y_pred)  # Fixed: removed [0] indexing


            # Log metrics for all models
            mlflow.log_metric("rmse", current_rmse)
            mlflow.log_metric("r2", current_r2)

            # Log hyperparameters for all models
            mlflow.log_params(grid_search.best_params_)

            # Keep track of the best model
            if current_rmse < best_rmse and current_r2 > best_r2:
                best_model = model
                best_rmse = current_rmse
                best_r2 = current_r2
                best_model_name = name

            results[name] = {
                'best_params': grid_search.best_params_,
                'rmse': current_rmse,
                'r2': current_r2
            }

            print(f"{name}: RMSE={current_rmse:.4f}, R²={current_r2:.4f}")

    # If we have a best model, log it with metrics
    if best_model:
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Save the best model
        model_path = f"model/model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Log the best model's metrics
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("best_r2", best_r2)

        print(f"Best model: {best_model_name} with RMSE={best_rmse:.4f}, R²={best_r2:.4f}")

    return results

# Plotting RMSE vs R²
def plot_rmse_vs_r2(results: Dict[str, Dict[str, float]], filename: str = 'src/model/images/rmse_vs_r2.png') -> None:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    model_names = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]

    plt.figure(figsize=(10, 6))
    plt.scatter(rmse_values, r2_values, color='navy', edgecolors='black', s=80, alpha=0.85)

    # Annotate points with model names and add background boxes for clarity
    for i, model_name in enumerate(model_names):
        plt.annotate(
            model_name,
            (rmse_values[i], r2_values[i]),
            textcoords='offset points',
            xytext=(7, -10 if i % 2 == 0 else 10),  # alternate vertical offset for better separation
            ha='left',
            fontsize=10,
            fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7)
        )

    plt.xlabel('Root Mean Squared Error (RMSE) — Lower is Better', fontsize=12, fontweight='bold')
    plt.ylabel('Coefficient of Determination (R²) — Higher is Better', fontsize=12, fontweight='bold')
    plt.title('Regression Model Performance: RMSE vs R²', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Main function
def main(data_path='data/processed/data.csv'):
    # Initialize MLflow
    mlflow.set_experiment("Ride_Cost_Prediction")
    
    X_train, X_test, y_train, y_test = data_splitting(data_path)
    models, pipelines, params = initializing_models()
    results = train(pipelines, params, X_train, y_train, X_test, y_test)
    plot_rmse_vs_r2(results)

# Execute the main function
if __name__ == "__main__":
    main()