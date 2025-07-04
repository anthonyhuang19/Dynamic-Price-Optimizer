# 🚀 Dynamic Pricing Optimization Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-1.30.0-0194E2.svg?logo=mlflow)](https://mlflow.org/)
## 💻  System Design of Dynamic Pricing
<div align="center">
  <img src="data/diagram.png" width="100%">
</div>

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Training Process](#-training-process)
- [API Usage](#-api-usage)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Project Overview

An intelligent pricing system that leverages machine learning to optimize product pricing in real-time, balancing profitability and market competitiveness. The system combines:

- **Comparison Model** (CatBoost, XGBoost, MLP,etc)
- **Market-sensitive features** (demand elasticity, competitor pricing)
- **MLflow-powered experiment tracking**
- **Production-ready Flask API**

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Real-time Adaptation** | Adjusts prices based on live market data | Maximizes revenue opportunities |
| **Explainable AI** | SHAP values for pricing decisions | Transparent business insights |
| **Automated Retraining** | Scheduled model refresh | Maintains prediction accuracy |

## 🏗 Model Architecture

![Architecture Diagram](src/model/images/rmse_vs_r2.png)

**CatBoost** is a gradient boosting algorithm designed to handle categorical features without the need for extensive preprocessing. It builds decision trees sequentially, with each tree correcting the errors made by the previous one. This method makes **CatBoost** particularly effective for dynamic pricing, as it can efficiently learn from both continuous and categorical market data to predict optimal prices.

#### **CatBoost in Dynamic Pricing:**
- **Handles Categorical Data**: Natively processes categorical features like product categories, store locations, and more.
- **Robust and Accurate**: Reduces overfitting with advanced regularization and produces highly accurate predictions.
- **Efficient**: Handles large datasets and missing values with ease, making it suitable for real-time pricing applications.
<div align="center">
  <img src="data/catboost.png" width="100%">
</div>


## 💻 Installation
--
```bash
# Clone repository
git clone https://github.com/anthonyhuang19/Dynamic-Price-Optimizer.git
cd dynamic-pricing

# Create environment
conda create -n pricing python=3.9 -y
conda activate pricing

# Install dependencies
pip install -r requirements.txt

# Initialize MLflow tracking
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
````

## 💻 **Installation with Docker**

To set up the Dynamic Pricing Optimization Engine locally with Docker, follow these steps:

### 1. **Build Docker Image**

First, build the Docker image using your project:

```bash
docker build -t my-python-project .
```

Alternatively, you can pull a pre-built Python image:

```bash
docker pull python:3.9-slim
```
### 2. ** Prepare Your Data **
Run the data preprocessing script inside the Docker container:
```bash
docker build -t my-python-project . 
````
### 3. **Train Models**
Run the model training and evaluation script inside the Docker container:
```bash
docker run --rm my-python-project  
````
### 4. ** Launch API **
To launch the API with Uvicorn, use Docker to run the application:
```bash
uvicorn src.serving.app:app --reload  
````
Now your pricing optimization engine API will be live at http://localhost:5000.

# 📊 Results
## Model Performance Comparison

| Model          | R² Score | RMSE | Training Time |
|----------------|----------|------|---------------|
| CatBoost       | 0.92     | 1.23 | 2m 15s        |
| Random Forest  | 0.89     | 1.45 | 1m 30s        |
| Neural Network | 0.87     | 1.52 | 3m 45s        |

## Performance Metrics

---

# 🔧 Training Process
### Configure training:
```yaml
# config/training_params.yaml
models:
  - catboost
  - random_forest

hyperparameters:
  catboost:
    iterations: 1000
    learning_rate: 0.03
  random_forest:
    n_estimators: 200
````

You can copy the above content and save it as `README.md` or any other file name with `.md` extension.

Let me know if you need further adjustments!
