import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom Standard Scaler for pandas DataFrames, compatible with sklearn pipelines.
    
    This scaler standardizes numeric columns by removing the mean and scaling to unit variance.
    Formula: (x - mean) / std
    
    Parameters
    -----
    columns : list, optional
        List of column names to scale. If None, automatically detects all
        numeric columns.
    
    Attributes
    -----
    mean_ : dict
        Dictionary mapping column names to mean values learned during fit.
    std_ : dict
        Dictionary mapping column names to standard deviation values learned during fit.
    feature_names_out_ : list
        List of output feature names after scaling.
    columns_ : list
        List of columns that will be scaled (determined during fit, excludes constant columns).
    """
    
    def __init__(self, columns=None):
        """
        Initialize the CustomStandardScaler.
        
        Parameters
        -----
        columns : list, optional
            List of column names to scale. If None, auto-detect numeric columns.
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        """
        Learn the mean and standard deviation from the training data.
        
        Parameters
        -----
        X : pd.DataFrame
            Training data containing numeric columns to scale.
        y : None
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        self : CustomStandardScaler
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Determine which columns to scale
        if self.columns is None:
            # Auto-detect numeric columns
            self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
            if not self.columns_:
                raise ValueError("No numeric columns found in DataFrame")
        else:
            # Validate specified columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
            # Validate that specified columns are numeric
            non_numeric_cols = [col for col in self.columns 
                              if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric_cols:
                raise ValueError(f"Columns are not numeric: {non_numeric_cols}")
            
            self.columns_ = self.columns
        
        # Calculate mean and std for each column
        self.mean_ = {}
        self.std_ = {}
        self.feature_names_out_ = []
        
        for col in self.columns_:
            col_data = X[col].dropna()  # Exclude NaN for calculation
            
            if len(col_data) == 0:
                raise ValueError(f"Column '{col}' has no valid numeric values")
            
            mean_val = col_data.mean()
            std_val = col_data.std()
            
            # Skip constant columns (std = 0)
            if std_val == 0 or np.isnan(std_val):
                continue
            
            self.mean_[col] = mean_val
            self.std_[col] = std_val
            self.feature_names_out_.append(col)
        
        # Update columns_ to exclude constant columns
        self.columns_ = list(self.mean_.keys())
        
        if not self.columns_:
            raise ValueError("No columns to scale (all columns are constant)")
        
        return self
    
    def transform(self, X):
        """
        Transform the data using standardization.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with scaled columns replacing original columns.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'mean_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index and copy original data
        result = X.copy()
        
        # Apply standardization to each column
        for col in self.columns_:
            mean_val = self.mean_[col]
            std_val = self.std_[col]
            
            # Apply standardization: (x - mean) / std
            result[col] = (X[col] - mean_val) / std_val
        
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


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom Min-Max Scaler for pandas DataFrames, compatible with sklearn pipelines.
    
    This scaler transforms numeric columns by scaling each feature to a given range [0, 1].
    Formula: (x - min) / (max - min)
    
    Parameters
    -----
    columns : list, optional
        List of column names to scale. If None, automatically detects all
        numeric columns.
    
    Attributes
    -----
    min_ : dict
        Dictionary mapping column names to minimum values learned during fit.
    max_ : dict
        Dictionary mapping column names to maximum values learned during fit.
    feature_names_out_ : list
        List of output feature names after scaling.
    columns_ : list
        List of columns that will be scaled (determined during fit, excludes constant columns).
    """
    
    def __init__(self, columns=None):
        """
        Initialize the CustomMinMaxScaler.
        
        Parameters
        -----
        columns : list, optional
            List of column names to scale. If None, auto-detect numeric columns.
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        """
        Learn the minimum and maximum values from the training data.
        
        Parameters
        -----
        X : pd.DataFrame
            Training data containing numeric columns to scale.
        y : None
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        self : CustomMinMaxScaler
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Determine which columns to scale
        if self.columns is None:
            # Auto-detect numeric columns
            self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
            if not self.columns_:
                raise ValueError("No numeric columns found in DataFrame")
        else:
            # Validate specified columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
            # Validate that specified columns are numeric
            non_numeric_cols = [col for col in self.columns 
                              if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric_cols:
                raise ValueError(f"Columns are not numeric: {non_numeric_cols}")
            
            self.columns_ = self.columns
        
        # Calculate min and max for each column
        self.min_ = {}
        self.max_ = {}
        self.feature_names_out_ = []
        
        for col in self.columns_:
            col_data = X[col].dropna()  # Exclude NaN for calculation
            
            if len(col_data) == 0:
                raise ValueError(f"Column '{col}' has no valid numeric values")
            
            min_val = col_data.min()
            max_val = col_data.max()
            
            # Skip constant columns (max - min = 0)
            if max_val == min_val or np.isnan(min_val) or np.isnan(max_val):
                continue
            
            self.min_[col] = min_val
            self.max_[col] = max_val
            self.feature_names_out_.append(col)
        
        # Update columns_ to exclude constant columns
        self.columns_ = list(self.min_.keys())
        
        if not self.columns_:
            raise ValueError("No columns to scale (all columns are constant)")
        
        return self
    
    def transform(self, X):
        """
        Transform the data using Min-Max scaling.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with scaled columns replacing original columns.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'min_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index and copy original data
        result = X.copy()
        
        # Apply Min-Max scaling to each column
        for col in self.columns_:
            min_val = self.min_[col]
            max_val = self.max_[col]
            
            # Apply Min-Max scaling: (x - min) / (max - min)
            result[col] = (X[col] - min_val) / (max_val - min_val)
        
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