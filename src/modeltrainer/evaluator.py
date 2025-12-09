import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .logger import setup_logger

logger = setup_logger("Evaluator")


class ModelEvaluator:
    """
    Handles the calculation and logging of regression model evaluation metrics.
    """

    def __init__(self):
        self.evaluation_results = {
            "RandomForestRegressor": {},
            "XGBRegressor": {},
            "LGBMRegressor": {},
            "CatBoostRegressor": {},
            "ElasticNet": {},
        }

    def evaluate(self, y_true, y_pred, model_name="RandomForestRegressor"):
        """
        Calculates RMSE, MAE, and R2 scores for a given model's predictions.

        Parameters
        ---
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values as returned by the model.
        model_name : str, optional
            Name of the model being evaluated. Defaults to "RandomForestRegressor".

        Returns
        ---
        dict
            A dictionary containing 'RMSE', 'MAE', and 'R2' scores.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.evaluation_results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

        logger.info(f"--- Evaluation Results for {model_name} ---")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE : {mae:.4f}")
        logger.info(f"R2  : {r2:.4f}")

        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def best_model(self, metric="RMSE", higher_better=False):
        """
        Identifies the best performing model based on a specific metric from stored results.

        Parameters
        ---
        metric : str
            The metric to use for comparison ("RMSE", "MAE", or "R2").
        higher_better : bool
            If True, a higher score is considered better (e.g., R2).
            If False, a lower score is considered better (e.g., RMSE, MAE).

        Returns
        ---
        tuple
            A tuple containing (best_model_name, best_score).

        Raises
        ---
        ValueError
            If the provided metric is not 'RMSE', 'MAE', or 'R2'.
        """
        if metric not in ["RMSE", "MAE", "R2"]:
            raise ValueError("Metric must be 'RMSE', 'MAE', or 'R2'.")

        best_model_name = None
        best_score = None

        for model_name, metrics in self.evaluation_results.items():
            score = metrics.get(metric)
            if score is None:
                continue
            if best_score is None:
                best_score = score
                best_model_name = model_name
            else:
                if higher_better:
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                else:
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
        if best_model_name:
            logger.info(
                f"Selected Best Model based on {metric}: {best_model_name} (Score = {best_score:.4f})"
            )
        return best_model_name, best_score

    def save_metrics(self, filepath):
        """
        Saves the evaluation metrics of all models to a JSON file.

        Parameters
        ---
        filepath : str
            Destination path for the JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(self.evaluation_results, f, indent=4)

        logger.info(f"Saved all evaluation metrics to {filepath}")
