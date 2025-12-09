import pandas as pd
from sklearn.model_selection import train_test_split
from .logger import setup_logger

logger = setup_logger("DataHandler")

class DataHandler:
    """
    Manages data loading, preparation, and splitting for model training.
    """

    def __init__(self, random_state=42):
        """
        Initializes the DataHandler with a specific random seed.

        Parameters
        ---
        random_state : int, optional
            Seed used by the random number generator for reproducible splits. Defaults to 42.
        """
        self.X, self.y = None, None
        self.random_state = random_state
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_and_split_data(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2):
        """
        Loads the dataset from a DataFrame, separates the target variable, 
        and splits it into training and testing sets.

        Parameters
        ---
        data : pd.DataFrame
            The complete input dataset.
        target_column : str
            The name of the column to be used as the target (label).
        test_size : float, optional
            The proportion of the dataset to include in the test split (e.g., 0.2 for 20%). Defaults to 0.2.

        Returns
        ---
        tuple
            A tuple containing four elements:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training target values.
            - y_test (pd.Series): Testing target values.
        """
        logger.info(f"Loading data... Original shape: {data.shape}")
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        logger.info(
            f"Data split completed. Train size: {self.X_train.shape[0]}, Test size: {self.X_test.shape[0]}"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
