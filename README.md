![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Final-orange)

# Python for Data Science - 23KDL1
STUDENT ID | FULL NAME | MAIN ROLE
---------- | --------- | -----
23280040   | Nguyen Duc Bao  | Modeling + Visualization
23280047   | To Truong Dong  | Datapreprocessing + Report
23280058   | Nguyen Duc Hieu | Modeling + Pipeline

# Final Term Project - Vietnam Housing prediction
Predict housing price in Vietnam using 5 distinct ML models:
- Linear-based model:
    - Elastic Net
- Tree-based models
    - RandomForest
    - XGBoost
    - CatBoost
    - LightGBM

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
- Subject: Real Estate
- Purpose: This Final Term Project aims to predict housing prices in Vietnam using machine learning based on available property information such as:
    - Address
    - Area, frontage, number of bathrooms, bedrooms, floors
    - Legal status, furniture state, etc.
- Goal: The project evaluates and compares the performance of multiple machine learning models, including **ElasticNet**, **RandomForest**, **XGBoost**, **CatBoost**, and **LightGBM**.
It also provides a complete and structured pipeline for data preprocessing, model training, evaluation, and visualization of results.

## Features
- Clean and preprocess raw housing data for modeling.
- Feature engineering to improve model performance.
- Train multiple machine learning models and tune hyperparameters.
- Evaluate models using metrics like RMSE, MAE, and R².
- Visualize feature importance, prediction errors, and model comparisons.
- Easily extendable pipeline for adding new models or datasets.

## Tech Stack

- Python: 3.10+

### 1. Data Handling & Manipulation

- NumPy – numerical computations

- Pandas – data manipulation and analysis

- Pathlib, os, datetime, json – file/path operations and date-time utilities

### 2. Data Preprocessing & Utilities

- re – regular expressions for text cleaning

- scipy – mathematical functions & statistical utilities

- argparse, configparser – command-line arguments & configuration management

- logging – system logging and debugging

- joblib – fast model serialization

### 3. Machine Learning & Modeling

- Scikit-learn – preprocessing, metrics (MSE, R², MAE), ElasticNet, and pipeline tools

- XGBoost, LightGBM, CatBoost – gradient boosting models

### 4. Hyperparameter Optimization

- Optuna – automated hyperparameter tuning and search

### 5. Visualization & Model Explainability

- Matplotlib, Seaborn – statistical plots and charts

- SHAP – model interpretability (local & global explanations)

- sklearn.inspection.PartialDependenceDisplay – Partial Dependence Plots (PDP)

### 6. Development Environment

- VS Code, Jupyter Notebook, virtualenv – development, analysis, and isolated environments

## Dataset

- **The dataset is taken from Kaggle**:  
[House Price Prediction Dataset Vietnam - 2024](https://tinyurl.com/vietnamhousepriceprediction).

- **You can also download it using Kaggle API**:
    ```bash
    kaggle datasets download nguyentiennhan/vietnam-housing-dataset-2024
    ```
## Project structure

    FinalTermProject_Python4DS/
    ├── config
    ├── data
    ├── src/
    │   ├── datapreprocessor/
    │   │   ├── __init__.py
    │   │   ├── data_preprocessing.py
    │   │   ├── encoder.py
    │   │   ├── imputer.py
    │   │   └── scaler.py
    │   ├── modeltrainer/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   ├── data_handler.py
    │   │   ├── evaluator.py
    │   │   ├── hyperparams_config.py
    │   │   ├── logger.py
    │   │   ├── model_trainer.py
    │   │   └── pipeline_factory.py 
    │   ├── visualizer/
    │   │   ├── __init__.py
    │   │   ├── plots.py
    │   │   ├── utils_layout.py
    │   │   ├── utils_style.py
    │   │   └── utils_validation.py
    │   └── main.py
    ├── notebooks/
    │   ├── eda.ipynb
    │   └── visualization.ipynb
    ├── LICENSE
    ├── README.md
    └── requirements.txt

## Installation

1. **Clone the repository.**
    ```bash
    git clone https://github.com/ducbao210/FinalTermProject_Python4DS.git
    cd FinalTermProject_Python4DS
    ```
2. Create a virtual environment (recommended).
    ```bash
    python -m venv venv 
    ```
    - Activate environment
        - Windows
        ```bash
        venv\Scripts\activate 
        ```
        - Linux/Mac
        ```bash
        source venv/bin/activate
        ```

3. Install required packages.
    ```bash
    pip install -r requirements.txt
    ```
4. Download the dataset from Kaggle and replace it in the data/raw folder.
    - See instruction in [Dataset](#dataset). Next, change dir (if needed), for instance:
        - Window
        ```bash
        move "C:/path/to/downloaded/file.txt" "D:/path/to/project/data/raw" 
        ```
        - Linux/Mac
        ```bash
        mv /path/to/downloaded/file.txt /path/to/project/data/raw/
        ```

## Usage:
1. Change direction to src
    ```bash
    cd src
    ```
2. Run with default configuration.
    ```bash
    python main.py
    ```
3. Run with custom arguments.
    ```bash
    python main.py --data_path <path> --model <[MODEL_NAMES]> --trials <INT> --test_size <FLOAT> --random_state <INT>
    ```
- *Example*:
    - Run specific models:
    ```bash
    python main.py --model XGBRegressor RandomForestRegressor
    ```
    - Full command example:
    ```bash
    python main.py \
    --data_path data/raw/house.csv \
    --model RandomForestRegressor CatBoostRegressor \
    --trials 120 \
    --test_size 0.2 \
    --random_state 123
    ```
## Results
```bash
- Test size: 0.36
- Random state: 8386
- Optuna trials: 50
```
After running the full ML pipeline with Optuna hyperparameters tuning(50 trials for each model), we obtained the following performance metrics on the test set.

Model | RMSE | MAE | R<sup>2</sup>
------|------|-----|--------------
LGBMRegressor | 1.3808315156649644 | 1.039682474444471 | 0.6130242041052543
XGBRegressor | 1.3870344665186412 | 1.0411404421398944 | 0.6095396611780065
RandomForestRegressor | 1.472791108940537 | 1.1289327350162746 | 0.5597648368979269
ElasticNet | 1.7051339806047927 | 1.3498374039372176 | 0.40990839068308094
CatBoost | 1.3843300639709741 | 1.0455688110540604 | 0.6110607949600845
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
