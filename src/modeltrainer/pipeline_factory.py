from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from datapreprocessor.encoder import CustomOneHotEncoder, CustomTargetEncoder
from datapreprocessor.imputer import CustomImputer
from datapreprocessor.scaler import CustomStandardScaler


class PipelineFactory:
    """
    Factory class to create Scikit-learn pipelines with automated preprocessing.
    """

    def __init__(self, features_config):
        """
        Initializes the factory with feature configurations.

        Parameters
        ---
        features_config : dict
            Configuration dictionary containing column lists.
            Expected keys: 'numeric_cols', 'onehot_cols', 'target_cols'.
        """
        self.features_config = features_config

    def create_pipeline(self, model):
        """
        Creates a complete pipeline including preprocessing and the model.

        Preprocessing steps:
        - Numeric: Median imputation and Standard Scaling.
        - Categorical (OneHot): Constant imputation ('missing') and OneHotEncoding.
        - Categorical (Target): Constant imputation ('missing') and TargetEncoding.

        Parameters
        ---
        model : sklearn.base.BaseEstimator
            The machine learning model instance to be added to the pipeline.

        Returns
        ---
        sklearn.pipeline.Pipeline
            The constructed pipeline.
        """
        numeric_transformer = Pipeline(
            [
                ("imputer", CustomImputer(numeric_strategy="median")),
                ("scaler", CustomStandardScaler()),
            ]
        )

        onehot_transformer = Pipeline(
            [
                (
                    "imputer",
                    CustomImputer(
                        categorical_strategy="constant",
                        categorical_fill_value="missing",
                    ),
                ),
                (
                    "onehot",
                    CustomOneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        target_transformer = Pipeline(
            [
                (
                    "imputer",
                    CustomImputer(
                        categorical_strategy="constant",
                        categorical_fill_value="missing",
                    ),
                ),
                (
                    "target",
                    CustomTargetEncoder(
                        target_type="continuous", smooth=10.0, columns=None
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    self.features_config.get("numeric_cols", []),
                ),
                (
                    "cat_onehot",
                    onehot_transformer,
                    self.features_config.get("onehot_cols", []),
                ),
                (
                    "target_enc",
                    target_transformer,
                    self.features_config.get("target_cols", []),
                ),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        return pipeline
