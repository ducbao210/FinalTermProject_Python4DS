import optuna, joblib, json
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import separated modules
from .data_handler import DataHandler
from .pipeline_factory import PipelineFactory
from .hyperparams_config import HyperparameterConfig
from .evaluator import ModelEvaluator
from .logger import setup_logger

logger = setup_logger("ModelTrainer")


class ModelTrainer:
    """
    Manages the end-to-end process of data loading, hyperparameter optimization,
    model training, and evaluation.
    """

    def __init__(self, random_state=42):
        """
        Initializes the ModelTrainer with necessary components and configurations.

        Sets up the data handler, hyperparameter configuration, and evaluator.
        It also initializes containers for storing the best optimized models
        and their corresponding parameters.

        Parameters
        ---
        random_state : int, optional
            Seed for reproducibility across data splitting and model training. Defaults to 42.
        """
        self.random_state = random_state
        self.data_manager = DataHandler(random_state)
        self.hp_config = HyperparameterConfig()
        self.evaluator = ModelEvaluator()

        self.best_models = {}
        self.best_params = {}

        # Mapping string -> Class
        self.model_classes = {
            "RandomForestRegressor": RandomForestRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor,
            "CatBoostRegressor": CatBoostRegressor,
            "ElasticNet": ElasticNet,
        }

    def load_and_split(self, df, target_col, test_size=0.2):
        """
        Loads the dataframe and splits it into training and testing sets.

        Parameters
        ---
        df : pd.DataFrame
            The raw dataframe.
        target_col : str
            The name of the target variable column.
        test_size : float, optional
            Proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns
        ---
        tuple
            Returns X_train, X_test, y_train, y_test from the DataHandler.
        """
        return self.data_manager.load_and_split_data(df, target_col, test_size)

    def optimize(self, model_name, features_config, n_trials=20, **fixed_params):
        """
        Executes the hyperparameter optimization process using Optuna.

        This method defines the objective function, runs the study, and retrains
        the model with the best parameters found.

        Parameters
        ---
        model_name : str
            The name of the model to optimize (must be in self.model_classes).
        features_config : dict
            Configuration for feature columns (numeric, categorical).
        n_trials : int, optional
            Number of Optuna trials to run. Defaults to 20.
        **fixed_params : dict
            Parameters to fix during optimization (passed to get_model_params).

        Raises
        ---
        ValueError
            If the model_name is not supported.
        """
        if model_name not in self.model_classes:
            logger.error(f"Model {model_name} not supported.")
            raise ValueError(f"Model {model_name} not supported.")

        logger.info(f"--- Starting Optimization for {model_name} ---")

        # Initialize the Pipeline
        pipeline_factory = PipelineFactory(features_config)

        def objective(trial):
            """
            The objective function for Optuna optimization.

            It builds a pipeline with hyperparameters suggested by the trial,
            performs K-Fold cross-validation, and returns the average RMSE
            score to be minimized.

            Parameters
            ---
            trial : optuna.trial.Trial
                The trial object used to suggest hyperparameters.

            Returns
            ---
            float
                The mean RMSE score from cross-validation (positive value).
            """
            # 1. Retrieve hyperparameters from HyperparameterConfig
            params = self.hp_config.get_model_params(trial, model_name, **fixed_params)

            # 2. Instantiate the model
            model_cls = self.model_classes[model_name]
            model = model_cls(**params)

            # 3. Create the pipeline using PipelineFactory
            pipeline = pipeline_factory.create_pipeline(model)

            # 4. Perform Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # Note: Accessing X_train and y_train directly from data_manager
            scores = cross_val_score(
                pipeline,
                self.data_manager.X_train,
                self.data_manager.y_train,
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            # Convert negative RMSE to positive RMSE for minimization
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params[model_name] = study.best_params
        logger.info(
            f"Optimization Finished. Best params for {model_name}: {study.best_params}"
        )

        # Refit the best model on the full training set
        self._fit_final_model(model_name, features_config)

    def _fit_final_model(self, model_name, features_config):
        """
        Retrains the model on the full training dataset using the best hyperparameters.

        Parameters
        ---
        model_name : str
            Name of the model to retrain.
        features_config : dict
            Feature configuration for the pipeline.
        """
        print(f"Retraining {model_name} on full training set...")

        best_params = self.best_params[model_name]
        model = self.model_classes[model_name](**best_params)

        pipeline_factory = PipelineFactory(features_config)
        final_pipeline = pipeline_factory.create_pipeline(model)

        final_pipeline.fit(self.data_manager.X_train, self.data_manager.y_train)
        self.best_models[model_name] = final_pipeline
        logger.info(f"Retraining completed for {model_name}")

    def evaluate_model(self, model_name):
        """
        Evaluates the optimized model on the test dataset.

        Parameters
        ---
        model_name : str
            Name of the model to evaluate.

        Returns
        ---
        dict
            Evaluation metrics (RMSE, MAE, R2) or None if model is not optimized.
        """
        if model_name not in self.best_models:
            print("Model not optimized yet.")
            return

        model = self.best_models[model_name]
        y_pred = model.predict(self.data_manager.X_test)

        return self.evaluator.evaluate(self.data_manager.y_test, y_pred, model_name)

    def save_model(self, model_name, filepath):
        """
        Saves the trained model pipeline to a file using joblib.

        Parameters
        ---
        model_name : str
            Name of the model to save.
        filepath : str
            Destination path for the model file.
        """
        if model_name in self.best_models:
            joblib.dump(self.best_models[model_name], filepath)
            logger.info(f"Saved {model_name} successfully to {filepath}")
        else:
            logger.warning(f"Cannot save {model_name}: Model not optimized yet.")

    def save_best_params(self, model_name, filepath):
        """
        Save the best hyperparameters to a JSON file.

        Parameters
        ---
        model_name : str
            Name of the model whose parameters to save.
        filepath : str
            Destination path for the JSON file.
        """
        if model_name in self.best_params:
            with open(filepath, "w") as f:
                json.dump(self.best_params[model_name], f, indent=4)
            logger.info(f"Saved best parameters for {model_name} to {filepath}")
        else:
            logger.warning(
                f"Cannot save parameters for {model_name}: No parameters found."
            )

    def save_X_train(self, model_name, filepath):
        """
        Saves X_train to a CSV file.

        Parameters
        ---
        model_name : str
            Name of the model (used for logging checks).
        filepath : str
            Full path to save the file.
        """
        if self.data_manager.X_train is not None:
            try:
                # Removed replace(".csv"...) logic since main.py already specifies the file name.
                # Old logic caused issues if input file doesn't have .csv extension or already has model name.

                self.data_manager.X_train.to_csv(filepath, index=False)
                logger.info(f"Saved X_train successfully to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save X_train: {e}")
                raise e
        else:
            logger.warning("Cannot save X_train: Data not loaded or split yet.")
