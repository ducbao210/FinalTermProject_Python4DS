import argparse
import pandas as pd
import os, shutil
from pathlib import Path
from datapreprocessor import DataPreprocessor

from datetime import datetime

from modeltrainer import settings
from modeltrainer import ModelTrainer
from modeltrainer import setup_logger

logger = setup_logger("MainProcess")


def reset_folder(path):
    """
    Create directory if it doesn't exist, or clear all contents if it exists.

    If the directory exists and contains files or subdirectories, all contents
    will be deleted. If it doesn't exist, a new empty directory will be created.

    Parameters
    ---
    path : str
        Directory path.
    """
    # If folder exists -> delete all contents inside
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create empty folder again
    os.makedirs(path, exist_ok=True)


def parse_arguments():
    """
    Parse and process command line arguments.

    This function uses the `argparse` library to define and collect configuration
    options from users through the terminal, allowing flexible runtime parameter
    changes without modifying source code.

    Parameters
    ---
    --data_path : str
        Path to input data file (CSV or Excel format).
    --model : list[str]
        List of specific model names to train (e.g., RandomForestRegressor XGBRegressor).
        Allows entering multiple models separated by spaces.
    --trials : int
        Maximum number of trials for hyperparameter optimization with Optuna.
    --test_size : float
        Test set size ratio. For example: 0.2 corresponds to 20%.
    --random_state : int
        Random seed to ensure result reproducibility.

    Returns
    ---
    argparse.Namespace
        Object containing parsed argument values.
        Access via: args.data_path, args.model, etc.
    """
    parser = argparse.ArgumentParser(description="Automated Model Training Pipeline")

    parser.add_argument("--data_path", type=str, help="Path to dataset csv/excel")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Model name (e.g., RandomForestRegressor, XGBRegressor,...)",
    )
    parser.add_argument("--trials", type=int, help="Number of Optuna trials")
    parser.add_argument(
        "--test_size", type=float, help="Test set size ratio (e.g., 0.2)"
    )
    parser.add_argument("--random_state", type=int, help="Random seed")

    return parser.parse_args()


def main():
    # Load and Update Config
    if settings is None:
        logger.error("Failed to initialize configuration. Exiting...")
        return

    args = parse_arguments()
    logger.info("Updating configuration from arguments...")
    settings.update_from_args(args)

    # Take from settings
    DATA_PATH = settings.DATA_PATH
    DEFAULT_MODEL = settings.DEFAULT_MODEL
    TEST_SIZE = settings.TEST_SIZE
    RANDOM_STATE = settings.RANDOM_STATE
    N_TRIALS = settings.N_TRIALS

    logger.info(f"--- CONFIGURATION ---")
    logger.info(f"Data Path: {DATA_PATH}")
    logger.info(f"Test Size: {TEST_SIZE}")
    logger.info(f"Random State: {RANDOM_STATE}")
    logger.info(f"Trials for Optuna: {N_TRIALS}")
    logger.info(f"----------------------")

    # Reading data
    logger.info("Reading data file...")
    try:
        df = DataPreprocessor.load_file(file_path=DATA_PATH, file_type="csv")
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        return

    # Preprocessing Raw Data
    logger.info("Performing raw data preprocessing...")
    df = DataPreprocessor.clean_name(df)
    df = DataPreprocessor.detect_outliers(df)
    logger.info(
        f"Actual columns after name cleaning and detecting outliers: {df.columns.tolist()}"
    )

    target_split_col = "address"
    if target_split_col in df.columns:
        logger.info(f"Splitting column '{target_split_col}'...")
        df = DataPreprocessor.split_column(
            df,
            column=target_split_col,
            max_columns=2,
            column_prefix="addr",
            reverse_split=True,
        )
        df["addr_1"] = df["addr_1"].fillna("")
        df["addr_2"] = df["addr_2"].fillna("")

        logger.info("Creating new columns 'new_address' contains 'District City'...")
        df["new_address"] = df["addr_2"] + ", " + df["addr_1"]
        df.drop(columns=["addr_1", "addr_2"], inplace=True)

        df = DataPreprocessor.split_column(
            df,
            column=target_split_col,
            max_columns=2,
            column_prefix="addr",
            reverse_split=False,
        )
        df["condo_complex"] = df["addr_1"].apply(
            lambda x: 1 if "Dự án" in str(x) else 0
        )
        logger.info("Creating new column 'condo_complex' for project identification")

        df.drop(columns=[target_split_col, "addr_1", "addr_2"], inplace=True)
    else:
        logger.warning(f"Column '{target_split_col}' not found to split.")

    # Save preprocessed file
    cleaned_data_path = "data/cleaned_dataset.csv"
    logger.info(f"Saving cleaned data to {cleaned_data_path}")
    DataPreprocessor.save_to_file(df, cleaned_data_path)

    # Define Feature Config
    FEATURES_CONFIG = {
        "numeric_cols": ["area", "frontage", "bedrooms", "floors", "condo_complex"],
        "onehot_cols": ["legal_status", "furniture_state"],
        "target_cols": ["new_address", "bathrooms", "bedrooms"],
    }
    TARGET_COL = "price"

    # Filter Columns
    logger.info("Filtering dataset to selected features only...")
    selected_features = (
        FEATURES_CONFIG["numeric_cols"]
        + FEATURES_CONFIG["onehot_cols"]
        + FEATURES_CONFIG["target_cols"]
    )
    all_needed_cols = selected_features + [TARGET_COL]

    # Dataframe input the model
    try:
        df = df[list(set(all_needed_cols))]
        logger.info(f"Filtered data shape: {df.shape}")
        logger.info(f"Columns used: {df.columns.tolist()}")
    except KeyError as e:
        logger.error(f"Error: Some features not exist in DataFrame: {e}")
        return

    # Initialize Trainer
    trainer = ModelTrainer(random_state=RANDOM_STATE)

    # Load and Split Data
    logger.info("Splitting data...")
    trainer.load_and_split(df, target_col=TARGET_COL, test_size=TEST_SIZE)

    # Run Optimize (Training & Tuning)
    if args.model:
        models_to_run = args.model
        logger.info(f"User selected specific model: {models_to_run}")
    else:
        models_to_run = [
            "RandomForestRegressor",
            "XGBRegressor",
            "LGBMRegressor",
            "ElasticNet",
            "CatBoostRegressor",
        ]
        logger.info(
            f"No specific model selected. Running comparison for: {models_to_run}"
        )

    project_root = Path(__file__).parent.parent
    # Create folder to save models
    os.makedirs(project_root / "models", exist_ok=True)

    # Create folder to save metrics
    os.makedirs(project_root / "metrics", exist_ok=True)

    # Create folder to store X_train after each section
    reset_folder(project_root / "X_train_saver")

    # Create folder to store hyperparemeter for each models after each section
    os.makedirs(project_root / "optimized_hyperparameters", exist_ok=True)

    timestamp_global = datetime.now().strftime("%Y%m%d_%H%M")

    # Loop for training and evaluating each model
    for model_name in models_to_run:
        logger.info(f"========================================")
        logger.info(f"Start processing: {model_name}")
        logger.info(f"========================================")
        try:
            trainer.optimize(
                model_name=model_name,
                features_config=FEATURES_CONFIG,
                n_trials=N_TRIALS,
            )
            # Evaluate Model
            logger.info("Evaluating best model...")
            metrics = trainer.evaluate_model(model_name)
            print(f"\n>>> FINAL RESULTS FOR {model_name}: {metrics}\n")

            # Save hyperparameters
            hyperparams_save_path = (
                str(project_root) + f"/optimized_hyperparameters/{model_name}.json"
            )
            trainer.save_best_params(model_name, hyperparams_save_path)
            print(
                f"Saved hyperparameters for individual model: {hyperparams_save_path}"
            )

            # Save X_train
            X_train_save_path = str(project_root) + f"/X_train_saver/{model_name}.csv"
            trainer.save_X_train(model_name, X_train_save_path)
            print(f"Saved X_train for individual model: {X_train_save_path}")

            # Save model
            individual_save_path = (
                str(project_root) + f"/models/{model_name}_{timestamp_global}.joblib"
            )
            trainer.save_model(model_name, individual_save_path)
            print(f"Saved individual model: {individual_save_path}")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            continue

    # Save metrics of all model after training and optimizing
    metrics_path = str(project_root) + f"/metrics/all_models_{timestamp_global}.json"
    trainer.evaluator.save_metrics(metrics_path)

    # Comparing
    evaluate_metric = "R2"
    logger.info("========================================")
    logger.info("SELECTING BEST MODEL")
    logger.info("========================================")
    best_name, best_score = trainer.evaluator.best_model(
        metric=evaluate_metric, higher_better=True
    )

    if best_name:
        print(
            f"\n>>> WINNER: {best_name} with {evaluate_metric.upper()} = {best_score:.4f}\n"
        )
        total_models_run = len(models_to_run)
        models_dir = Path(__file__).parent.parent / "models"
        save_path = (
            str(models_dir)
            + f"/BEST_OF_{total_models_run}_MODELS_{best_name}_{timestamp_global}.joblib"
        )
        trainer.save_model(best_name, save_path)
        logger.info(f"Saved WINNER model to: {save_path}")
    else:
        logger.error("No models were successfully trained/evaluated.")

    logger.info("Pipeline Completed successfully.")
    logger.info("================================")


if __name__ == "__main__":
    main()
