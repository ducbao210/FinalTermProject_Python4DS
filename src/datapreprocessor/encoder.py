import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoder for pandas DataFrames, compatible with sklearn pipelines.
    
    This encoder transforms categorical columns into one-hot encoded columns,
    where each category becomes a binary column.
    
    Parameters
    -----
    columns : list, optional
        List of column names to encode. If None, automatically detects all
        categorical columns (object, category, or string dtype).
    handle_unknown : str, default 'ignore'
        How to handle unknown categories during transform:
        - 'ignore': Unknown categories are encoded as all zeros
        - 'error': Raise ValueError if unknown categories are encountered
    sparse_output : bool, default False
        If True, returns sparse DataFrame. If False, returns dense DataFrame.
    
    Attributes
    -----
    categories_ : dict
        Dictionary mapping column names to lists of unique categories learned during fit.
    feature_names_out_ : list
        List of output feature names after one-hot encoding.
    columns_ : list
        List of columns that will be encoded (determined during fit).
    """
    
    def __init__(self, columns=None, handle_unknown='ignore', sparse_output=False):
        """
        Initialize the CustomOneHotEncoder.
        
        Parameters
        -----
        columns : list, optional
            List of column names to encode. If None, auto-detect categorical columns.
        handle_unknown : str, default 'ignore'
            How to handle unknown categories: 'ignore' or 'error'.
        sparse_output : bool, default False
            If True, returns sparse DataFrame. If False, returns dense DataFrame.
        """
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        
        if handle_unknown not in ['ignore', 'error']:
            raise ValueError(f"handle_unknown must be 'ignore' or 'error', got {handle_unknown}")
    
    def fit(self, X, y=None):
        """
        Learn the categories from the training data.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Training data containing categorical columns to encode.
            If not a DataFrame, will be converted automatically.
        y : None
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        self : CustomOneHotEncoder
            Returns the instance itself.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Determine which columns to encode
        if self.columns is None:
            # Auto-detect categorical columns
            self.columns_ = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if not self.columns_:
                raise ValueError("No categorical columns found in DataFrame")
        else:
            # Validate specified columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.columns_ = self.columns
        
        # Learn categories for each column
        self.categories_ = {}
        self.feature_names_out_ = []
        
        for col in self.columns_:
            # Get unique categories, excluding NaN
            categories = X[col].dropna().unique().tolist()
            categories = sorted(categories)  # Sort for consistency
            
            if len(categories) == 0:
                raise ValueError(f"Column '{col}' has no valid categories")
            
            self.categories_[col] = categories
            
            # Generate feature names for this column
            for cat in categories:
                # Create feature name: column_name_category_value
                feature_name = f"{col}_{cat}"
                self.feature_names_out_.append(feature_name)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using one-hot encoding.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded columns. Original columns are removed.
            If sparse_output=True, returns sparse DataFrame.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'categories_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Collect all one-hot encoded columns in a dictionary first
        encoded_columns = {}
        
        # Process each column
        for col in self.columns_:
            col_data = X[col]
            
            # Handle unknown categories check first
            if self.handle_unknown == 'error':
                # Check for unknown categories
                known_cats = set(self.categories_[col])
                unique_vals = col_data.dropna().unique()
                unknown_cats = [val for val in unique_vals if val not in known_cats]
                
                if unknown_cats:
                    raise ValueError(f"Unknown categories found in column '{col}': {unknown_cats}")
            
            # Create one-hot encoded columns for each category
            for cat in self.categories_[col]:
                feature_name = f"{col}_{cat}"
                # Check if value equals category (handling NaN properly)
                encoded_columns[feature_name] = (col_data == cat).astype(int)
            
            # If handle_unknown == 'ignore', unknown categories are already encoded as all zeros
            # (since they don't match any known category)
        
        # Create result DataFrame by concatenating all columns at once
        result = pd.DataFrame(encoded_columns, index=X.index)
        
        # Convert to sparse DataFrame if requested
        if self.sparse_output:
            result = result.astype(pd.SparseDtype("int", fill_value=0))
        
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


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Label Encoder for pandas DataFrames, compatible with sklearn pipelines.
    
    This encoder transforms categorical columns into integer labels,
    where each category is assigned a unique integer (0, 1, 2, ...).
    Original categorical columns are removed after encoding.
    
    Parameters
    -----
    columns : list, optional
        List of column names to encode. If None, automatically detects all
        categorical columns (object, category, or string dtype).
    handle_unknown : str, default 'ignore'
        How to handle unknown categories during transform:
        - 'ignore': Unknown categories are encoded as unknown_value
        - 'error': Raise ValueError if unknown categories are encountered
    unknown_value : int, default -1
        Value to assign to unknown categories when handle_unknown='ignore'.
    
    Attributes
    -----
    label_mapping_ : dict
        Dictionary mapping column names to dictionaries of category->integer mappings.
    feature_names_out_ : list
        List of output feature names after label encoding.
    columns_ : list
        List of columns that will be encoded (determined during fit).
    """
    
    def __init__(self, columns=None, handle_unknown='ignore', unknown_value=-1):
        """
        Initialize the CustomLabelEncoder.
        
        Parameters
        -----
        columns : list, optional
            List of column names to encode. If None, auto-detect categorical columns.
        handle_unknown : str, default 'ignore'
            How to handle unknown categories: 'ignore' or 'error'.
        unknown_value : int, default -1
            Value to assign to unknown categories when handle_unknown='ignore'.
        """
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        
        if handle_unknown not in ['ignore', 'error']:
            raise ValueError(f"handle_unknown must be 'ignore' or 'error', got {handle_unknown}")
    
    def fit(self, X, y=None):
        """
        Learn the label mappings from the training data.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Training data containing categorical columns to encode.
            If not a DataFrame, will be converted automatically.
        y : None
            Ignored. Present for sklearn compatibility.
        
        Returns
        -------
        self : CustomLabelEncoder
            Returns the instance itself.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Determine which columns to encode
        if self.columns is None:
            # Auto-detect categorical columns
            self.columns_ = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if not self.columns_:
                raise ValueError("No categorical columns found in DataFrame")
        else:
            # Validate specified columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.columns_ = self.columns
        
        # Learn label mappings for each column
        self.label_mapping_ = {}
        self.feature_names_out_ = []
        
        for col in self.columns_:
            # Get unique categories, excluding NaN
            categories = X[col].dropna().unique().tolist()
            categories = sorted(categories)  # Sort for consistency
            
            if len(categories) == 0:
                raise ValueError(f"Column '{col}' has no valid categories")
            
            # Create mapping from category to integer (0, 1, 2, ...)
            label_map = {cat: idx for idx, cat in enumerate(categories)}
            self.label_mapping_[col] = label_map
            
            # Generate feature name for this column
            feature_name = f"{col}_encoded"
            self.feature_names_out_.append(feature_name)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using label encoding.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with original categorical columns removed and replaced by 
            encoded columns (with suffix "_encoded"). Other columns are preserved.
        """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'label_mapping_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index and copy original data
        result = X.copy()
        
        # Process each column
        for col in self.columns_:
            col_data = X[col]
            label_map = self.label_mapping_[col]
            encoded_col_name = f"{col}_encoded"
            
            # Check for unknown categories if handle_unknown == 'error'
            if self.handle_unknown == 'error':
                known_cats = set(label_map.keys())
                unique_vals = col_data.dropna().unique()
                unknown_cats = [val for val in unique_vals if val not in known_cats]
                
                if unknown_cats:
                    raise ValueError(f"Unknown categories found in column '{col}': {unknown_cats}")
            
            # Apply label encoding using vectorized map operation
            encoded_values = col_data.map(label_map)
            
            # Handle unknown categories (values not in label_map)
            if self.handle_unknown == 'ignore':
                # Fill NaN values (which include both actual NaN and unknown categories)
                # First, identify which NaN are actual NaN vs unknown categories
                mask_actual_nan = col_data.isna()
                mask_unknown = encoded_values.isna() & ~mask_actual_nan
                
                # Replace unknown categories with unknown_value
                encoded_values.loc[mask_unknown] = self.unknown_value
                # Keep actual NaN as NaN (already handled by map)
            
            # Add encoded column to result
            result[encoded_col_name] = encoded_values
        
        # Remove original categorical columns after encoding
        result = result.drop(columns=self.columns_)
        
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

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Target Encoder for pandas DataFrames, compatible with sklearn pipelines.
    
    This encoder transforms categorical columns using target encoding with smoothing.
    Formula: (n * mean{group} + m * mean{global}) / (n + m)
    where n is the count of the category and m is the smoothing parameter.
    
    Parameters
    -----
    columns : list
        List of column names to encode. Must be specified (cannot be None).
    smooth : float, default 10
        Smoothing parameter. Higher values give more weight to the global mean.
    
    Attributes
    -----
    mapping_ : dict
        Dictionary mapping column names to Series of smoothed mean values for each category.
    global_mean_ : float
        Global mean of the target variable learned during fit.
    feature_names_out_ : list
        List of output feature names after target encoding.
    columns_ : list
        List of columns that will be encoded.
    """
    
    def __init__(self, target_type="continuous", smooth=10, columns=None):
        """
        Args:
            target_type (str): Placeholder for compatibility with pipeline (ignored in logic).
            smooth (float): Smoothing parameter (m).
            columns (list): List of columns to encode. If None, encodes all passed columns.
        """
        self.target_type = target_type
        self.smooth = smooth
        self.columns = columns
        
        if not isinstance(self.smooth, (int, float)) or self.smooth < 0:
            raise ValueError("smooth must be a non-negative number")
    
    def fit(self, X, y):
        """
        Learn the target encoding mappings from the training data.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Training data containing categorical columns to encode.
            If not a DataFrame, will be converted automatically.
        y : array-like
            Target variable for computing target encoding.
        
        Formula: (n * mean{group} + m * mean{global}) / (n + m)
        
        Returns
        -------
        self : CustomTargetEncoder
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Convert y to Series if needed
        if y is None:
            raise ValueError("Target y is required for Target Encoding fit method.")
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        
        # Validate columns exist
        # If columns is None 
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            # Validate columns
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.columns_ = self.columns
        
        # Calculate global mean
        self.global_mean_ = y.mean()
        
        # Learn mappings for each column
        self.mapping_ = {}
        self.feature_names_out_ = []
        
        for col in self.columns_:
            # Group by category and calculate statistics
            stats = y.groupby(X[col]).agg(['count', 'mean'])
            
            # Formula: (n * mean_group + m * mean_global) / (n + m)
            smooth_mean = ((stats['count'] * stats['mean']) + 
                          (self.smooth * self.global_mean_)) / (stats['count'] + self.smooth)
            
            self.mapping_[col] = smooth_mean
            
            # Generate feature name (same as original column name for target encoding)
            self.feature_names_out_.append(col)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using target encoding.
        
        Parameters
        -----
        X : pd.DataFrame or array-like
            Data to transform. If not a DataFrame, will be converted automatically.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with target encoded columns replacing original columns.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if fit has been called
        if not hasattr(self, 'mapping_'):
            raise ValueError("This instance is not fitted, call method fit before")
        
        # Validate that all required columns exist
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index and copy original data
        result = X.copy()
        
        # Apply target encoding to each column
        for col in self.columns_:
            # Map categories to smoothed mean values
            # Unknown categories will be NaN, fill with global_mean
            if col in result.columns:
                result[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_)
            else:
                result[col] = self.global_mean_

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