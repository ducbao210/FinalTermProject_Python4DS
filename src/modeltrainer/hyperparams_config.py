class HyperparameterConfig:
    """
    Defines the hyperparameter search spaces for various regression models using Optuna.
    """

    def get_model_params(self, trial, model_name, **params):
        """
        Retrieves the hyperparameter search space for a specific model.

        Supports overriding parameters via **params.

        Parameters
        ---
        trial : optuna.trial.Trial
            The Optuna trial object used to suggest parameters.
        model_name : str
            Name of the model (e.g., "RandomForestRegressor", "XGBRegressor").
        **params : dict
            Arbitrary keyword arguments to override suggested parameters.

        Returns
        ---
        dict
            A dictionary of hyperparameters suitable for the specified model.
        """

        def suggest(name, func_type, *args, **kwargs):
            """
            Suggests a hyperparameter value or returns a fixed override.

            If the parameter 'name' is present in the 'params' kwargs (fixed_params),
            that value is returned directly, bypassing Optuna. Otherwise, Optuna's
            suggestion method (func_type) is called.

            Parameters
            ---
            name : str
                The name of the hyperparameter.
            func_type : str
                The Optuna method name (e.g., 'suggest_int', 'suggest_float').
            *args : tuple
                Positional arguments for the Optuna method.
            **kwargs : dict
                Keyword arguments for the Optuna method.

            Returns
            ---
            Any
                The value for the hyperparameter.
            """
            if name in params:
                return params[name]
            return getattr(trial, func_type)(name, *args, **kwargs)

        if model_name == "RandomForestRegressor":
            """
            Random Forest Hyperparameters:
            - n_estimators: Number of trees in the forest (100-500).
            - max_depth: Maximum depth of the tree (5-30).
            - min_samples_split: Min samples required to split an internal node (2-15).
            - min_samples_leaf: Min samples required to be at a leaf node (1-10).
            - max_features: Number of features to consider when looking for the best split.
            """
            return {
                "n_estimators": suggest("n_estimators", "suggest_int", 100, 500),
                "max_depth": suggest("max_depth", "suggest_int", 5, 30),
                "min_samples_split": suggest("min_samples_split", "suggest_int", 2, 15),
                "min_samples_leaf": suggest("min_samples_leaf", "suggest_int", 1, 10),
                "max_features": suggest(
                    "max_features", "suggest_categorical", ["sqrt", "log2"]
                ),
                "n_jobs": -1,
            }

        elif model_name == "XGBRegressor":
            """
            XGBoost Hyperparameters:
            - n_estimators: Number of gradient boosted trees (100-1000).
            - learning_rate: Step size shrinkage used in update to prevent overfitting (0.01-0.3).
            - max_depth: Maximum depth of a tree (3-10).
            - subsample: Subsample ratio of the training instances (0.6-1.0).
            - colsample_bytree: Subsample ratio of columns when constructing each tree (0.6-1.0).
            - reg_alpha: L1 regularization term on weights (0-10).
            - reg_lambda: L2 regularization term on weights (0-10).
            """
            return {
                "n_estimators": suggest("n_estimators", "suggest_int", 100, 1000),
                "learning_rate": suggest("learning_rate", "suggest_float", 0.01, 0.3),
                "max_depth": suggest("max_depth", "suggest_int", 3, 10),
                "subsample": suggest("subsample", "suggest_float", 0.6, 1.0),
                "colsample_bytree": suggest(
                    "colsample_bytree", "suggest_float", 0.6, 1.0
                ),
                "reg_alpha": suggest("reg_alpha", "suggest_float", 0, 10),
                "reg_lambda": suggest("reg_lambda", "suggest_float", 0, 10),
                "n_jobs": -1,
            }

        elif model_name == "LGBMRegressor":
            """
            LightGBM Hyperparameters:
            - n_estimators: Number of boosted trees (100-1000).
            - learning_rate: Boosting learning rate (0.01-0.3).
            - num_leaves: Max number of leaves in one tree (20-150).
            - max_depth: Max depth for tree model. -1 means no limit (-1 to 15).
            - subsample: Subsample ratio of the training instance (0.6-1.0).
            - colsample_bytree: Subsample ratio of columns when constructing each tree (0.6-1.0).
            - min_child_samples: Minimum number of data needed in a child (leaf) (10-100).
            """
            return {
                "n_estimators": suggest("n_estimators", "suggest_int", 100, 1000),
                "learning_rate": suggest("learning_rate", "suggest_float", 0.01, 0.3),
                "num_leaves": suggest("num_leaves", "suggest_int", 20, 150),
                "max_depth": suggest("max_depth", "suggest_int", -1, 15),
                "subsample": suggest("subsample", "suggest_float", 0.6, 1.0),
                "colsample_bytree": suggest(
                    "colsample_bytree", "suggest_float", 0.6, 1.0
                ),
                "min_child_samples": suggest(
                    "min_child_samples", "suggest_int", 10, 100
                ),
                "n_jobs": -1,
                "verbose": -1,
            }

        elif model_name == "CatBoostRegressor":
            """
            CatBoost Hyperparameters:
            - iterations: The maximum number of trees that can be built (500-1500).
            - learning_rate: The learning rate (0.01-0.3).
            - depth: Depth of the tree (4-10).
            - l2_leaf_reg: Coefficient at the L2 regularization term of the cost function (1-10).
            - border_count: The number of splits for numerical features (32-255).
            """
            return {
                "iterations": suggest("iterations", "suggest_int", 500, 1500),
                "learning_rate": suggest("learning_rate", "suggest_float", 0.01, 0.3),
                "depth": suggest("depth", "suggest_int", 4, 10),
                "l2_leaf_reg": suggest("l2_leaf_reg", "suggest_float", 1, 10),
                "border_count": suggest("border_count", "suggest_int", 32, 255),
                "thread_count": -1,
                "verbose": 0,
                "allow_writing_files": False,
            }

        elif model_name == "ElasticNet":
            """
            ElasticNet Hyperparameters:
            - alpha: Constant that multiplies the penalty terms (0.01-100.0, log scale).
            - l1_ratio: The ElasticNet mixing parameter (0.0-1.0).
                        0.0 corresponds to L2 (Ridge), 1.0 to L1 (Lasso).
            """
            return {
                "alpha": suggest("alpha", "suggest_float", 0.01, 100.0, log=True),
                "l1_ratio": suggest("l1_ratio", "suggest_float", 0.0, 1.0),
                "max_iter": 2000,
            }

        return {}
