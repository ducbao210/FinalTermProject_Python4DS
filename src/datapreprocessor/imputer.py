import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom Imputer for pandas DataFrames, compatible with sklearn pipelines.
    
    This imputer handles missing values with different strategies for numeric
    and categorical columns, using sklearn's SimpleImputer internally.
    
    Parameters
    -----
    numeric_strategy : str, default 'mean'
        Strategy for imputing numeric columns.
        Options: 'mean', 'median', 'most_frequent', 'constant'
    categorical_strategy : str, default 'most_frequent'
        Strategy for imputing categorical columns.
        Options: 'most_frequent', 'constant'
    numeric_fill_value : float or int, optional
        Fill value for numeric columns when numeric_strategy='constant'.
        If None and strategy='constant', defaults to 0.
    categorical_fill_value : str, optional
        Fill value for categorical columns when categorical_strategy='constant'.
        If None and strategy='constant', defaults to 'missing'.
    columns : list, optional
        List of column names to impute. If None, imputes all columns.
    
    Attributes
    -----
    numeric_imputer_ : SimpleImputer
        SimpleImputer instance for numeric columns.
    categorical_imputer_ : SimpleImputer
        SimpleImputer instance for categorical columns.
    numeric_columns_ : list
        List of numeric columns that will be imputed.
    categorical_columns_ : list
        List of categorical columns that will be imputed.
    feature_names_out_ : list
        List of output feature names after imputation.
    columns_ : list
        List of all columns that will be imputed.
    """
    
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent',
                 numeric_fill_value=None, categorical_fill_value=None, columns=None):
        """
        Initialize the CustomImputer.
        
        Parameters
        -----
        numeric_strategy : str, default 'mean'
            Strategy for numeric columns: 'mean', 'median', 'most_frequent', 'constant'
        categorical_strategy : str, default 'most_frequent'
            Strategy for categorical columns: 'most_frequent', 'constant'
        numeric_fill_value : float or int, optional
            Fill value for numeric columns when strategy='constant'
        categorical_fill_value : str, optional
            Fill value for categorical columns when strategy='constant'
        columns : list, optional
            List of column names to impute. If None, imputes all columns.
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_fill_value = categorical_fill_value
        self.columns = columns
        
        # Validate strategies
        valid_numeric_strategies = ['mean', 'median', 'most_frequent', 'constant']
        valid_categorical_strategies = ['most_frequent', 'constant']
        
        if numeric_strategy not in valid_numeric_strategies:
            raise ValueError(f"numeric_strategy must be one of {valid_numeric_strategies}, "
                           f"got {numeric_strategy}")
        
        if categorical_strategy not in valid_categorical_strategies:
            raise ValueError(f"categorical_strategy must be one of {valid_categorical_strategies}, "
                           f"got {categorical_strategy}")
        
        # Set default fill values if needed
        if numeric_strategy == 'constant' and numeric_fill_value is None:
            self.numeric_fill_value = 0
        
        if categorical_strategy == 'constant' and categorical_fill_value is None:
            self.categorical_fill_value = 'missing'
    
    def fit(self, X, y=None):
        """
        Learn the imputation parameters from the training data.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Training data containing columns to impute.
            If not a DataFrame, will be converted automatically.
        y : None
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        self : CustomImputer
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Determine which columns to impute
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            # Validate specified columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.columns_ = self.columns
        
        # Classify columns into numeric and categorical
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_columns_.append(col)
            else:
                self.categorical_columns_.append(col)
        
        # Create and fit numeric imputer
        if self.numeric_columns_:
            numeric_imputer_kwargs = {'strategy': self.numeric_strategy}
            if self.numeric_strategy == 'constant':
                numeric_imputer_kwargs['fill_value'] = self.numeric_fill_value
            
            self.numeric_imputer_ = SimpleImputer(**numeric_imputer_kwargs)
            self.numeric_imputer_.fit(X[self.numeric_columns_])
        else:
            self.numeric_imputer_ = None
        
        # Create and fit categorical imputer
        if self.categorical_columns_:
            categorical_imputer_kwargs = {'strategy': self.categorical_strategy}
            if self.categorical_strategy == 'constant':
                categorical_imputer_kwargs['fill_value'] = self.categorical_fill_value
            
            self.categorical_imputer_ = SimpleImputer(**categorical_imputer_kwargs)
            self.categorical_imputer_.fit(X[self.categorical_columns_])
        else:
            self.categorical_imputer_ = None
        
        # Store feature names
        self.feature_names_out_ = self.columns_.copy()
        
        return self
    
    def transform(self, X):
        """
        Transform the data by imputing missing values.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with imputed columns replacing original columns.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'numeric_imputer_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index and copy original data
        result = X.copy()
        
        # Apply numeric imputation
        if self.numeric_columns_ and self.numeric_imputer_ is not None:
            numeric_imputed = self.numeric_imputer_.transform(X[self.numeric_columns_])
            # Convert to DataFrame to preserve column names and index
            numeric_df = pd.DataFrame(
                numeric_imputed,
                columns=self.numeric_columns_,
                index=X.index
            )
            result[self.numeric_columns_] = numeric_df
        
        # Apply categorical imputation
        if self.categorical_columns_ and self.categorical_imputer_ is not None:
            categorical_imputed = self.categorical_imputer_.transform(X[self.categorical_columns_])
            # Convert to DataFrame to preserve column names and index
            categorical_df = pd.DataFrame(
                categorical_imputed,
                columns=self.categorical_columns_,
                index=X.index
            )
            result[self.categorical_columns_] = categorical_df
        
        return result
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters
        -----
        input_features : array-like of str, optional
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        feature_names_out : ndarray of str objects
            Array of output feature names.
        """
        if not hasattr(self, 'feature_names_out_'):
            raise ValueError("This instance is not fitted")
        return np.array(self.feature_names_out_)