import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest

class DataPreprocessor:
    """
    Utility class for simple data preprocessing include static methods
    such as load_file, clean_name, detect_outliers, create_datetime_features, save_to_file
    """
    @staticmethod
    def load_file(file_path: str, file_type: str = None, sheet_name='Sheet1',
                  json_format=None, encoding='utf-8', **kwargs) -> pd.DataFrame:
        """
        Load a file into a pandas DataFrame with support for CSV, Excel, and JSON formats.

        Parameters
        ---
        file_path : str
            The path to the file to load.
        file_type : str, optional
            Type of file: 'csv', 'xlsx', 'xls', or 'json'. If None, will try to infer from file extension.
        sheet_name : str, default 'Sheet1'
            Name of the sheet to load (only used for Excel files).
        json_format : str, optional
            Format for loading JSON files (passed as `orient` to pd.read_json). Defaults to 'records'.
        encoding : str, default 'utf-8'
            Encoding to use for reading files (primarily for CSV files).
        **kwargs
            Additional arguments passed to the relevant pandas file loading function.

        Returns
        ---
        pd.DataFrame
            The loaded data as a pandas DataFrame.
        """
        # Resolve relative paths from PROJECT root
        file_path = Path(file_path)
        
        # If path is relative, resolve it from PROJECT root
        if not file_path.is_absolute():
            # Get PROJECT root: from src/datapreprocessor/data_preprocessing.py go up 2 levels
            current_file_dir = Path(__file__).parent  # src/datapreprocessor/
            project_root = current_file_dir.parent.parent  # PROJECT/
            file_path = project_root / file_path
        
        file_path = file_path.resolve()

        if file_type is None:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_type = "csv"
            elif suffix in ["xlsx", "xls"]:
                file_type = suffix[1:]
            elif suffix == "json":
                file_type = "json"
            else:
                raise ValueError(f"Cannot auto-detect file type from extension: {suffix}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_type == "csv":
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            elif file_type == "json":
                orient = json_format if json_format is not None else "records"
                return pd.read_json(file_path, orient=orient, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}") from e


    @staticmethod
    def clean_name(df: pd.DataFrame, inplace: bool = False):
        """
        Clean all column names to snake_case format

        Parameters
        ---
        df : pd.DataFrame
            DataFrame with column to clean
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        ---
        pd.DataFrame
            DataFrame with cleaned column names if inplace=True, or modified DataFrame if inplace=False
        """
        columns_to_clean = df.columns.to_list()
        name_mapping = {}

        for col in columns_to_clean:
            col_str = str(col).strip()
            
            # Replace spaces and special characters with underscores
            cleaned = col_str.lower()
            cleaned = re.sub(r'[\s\-\.]+', '_', cleaned)
            cleaned = re.sub(r'[^a-z0-9_]', '_', cleaned)
            cleaned = re.sub(r'_+', '_', cleaned)
            cleaned = cleaned.strip('_')
            
            if not cleaned:
                cleaned = 'unnamed'
            
            if cleaned and cleaned[0].isdigit():
                cleaned = 'col_' + cleaned
            
            if cleaned != col:
                name_mapping[col] = cleaned
        
        if name_mapping:
            if inplace:
                df.rename(columns=name_mapping, inplace=True)
                return df
            else:
                return df.rename(columns=name_mapping)
        else:
            return df.copy() if not inplace else df

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr', threshold: float = None,
                        return_mask: bool = False, **kwargs):
        """
        Detect outliers in DataFrame using IQR, Z-score, or IsolationForest.
        
        Parameters
        ---
        df : pd.DataFrame
            DataFrame to analyze
        columns : list, optional
            List of column name to analyze. If None, analyzes all numeric columns
        method : str, default 'iqr'
            Method to detect outliers: 'iqr', 'zscore', or 'isolation_forest'
        threshold : float, optional
            - For 'iqr': multiplier for IQR (default: 1.5)
            - For 'zscore': z-score threshold (default: 3.0)
            - For 'isolation_forest': contamination rate (default: 0.1)
        return_mask : bool, default False
            If True, return boolean mask. If False, return DataFrame with outliers removed
        **kwargs : dict
            Additional parameters for IsolationForest
        
        Returns
        ---
        pd.DataFrame or pd.Series
            - If return_mask=True: boolean Series/DataFrame indicating outliers
            - If return_mask=False: DataFrame with outliers removed (or original if column specified)
        """
        if columns:
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Column '{col} is not numeric")
            columns_to_check = columns
            numeric_df = df[columns]
        else:
            columns_to_check = df.select_dtypes(include=[np.number]).columns.to_list()
            if not columns_to_check:
                raise ValueError("No numeric column found in DataFrame")
            numeric_df = df[columns_to_check]

        if threshold is None:
            threshold_map = {
                'iqr': 1.5,
                'zscore': 1.5,
                'isolation_forest': 0.1
            }
            threshold = threshold_map.get(method, 1.5)

        outlier_mask = pd.Series([False] * len(df), index=df.index)

        if method == 'iqr':
            for col in columns_to_check:
                Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit', axis=0))
            outlier_mask = (z_scores > threshold).any(axis=1)
        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=threshold, **kwargs)
            predictions = iso_forest.fit_predict(numeric_df)
            clean_mask = pd.Series(predictions == -1, index=numeric_df.index)
            outlier_mask.loc[numeric_df.index] = clean_mask
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'iqr', 'zscore' or 'isolation_forest'")

        return outlier_mask if return_mask else df[~outlier_mask]

    @staticmethod
    def create_datetime_features(df: pd.DataFrame, column: str,
                                 features: list = None, inplace: bool = False):
        """
        Create new features from a datetime column.
        
        Parameters
        ---
        df : pd.DataFrame
            DataFrame containing the datetime column
        column : str
            Column name containing datetime values
        features : list, optional
            List of features to extract. If None, extracts all available features.
            Available features:
            - 'year': Year (e.g., 2023)
            - 'month': Month (1-12)
            - 'day': Day of month (1-31)
            - 'hour': Hour (0-23)
            - 'minute': Minute (0-59)
            - 'second': Second (0-59)
            - 'day_of_week': Day of week (0=Monday, 6=Sunday)
            - 'day_of_year': Day of year (1-365/366)
            - 'week': Week number (1-52/53)
            - 'quarter': Quarter (1-4)
            - 'is_weekend': Boolean (True if Saturday or Sunday)
            - 'is_weekday': Boolean (True if Monday-Friday)
            - 'is_month_start': Boolean (True if first day of month)
            - 'is_month_end': Boolean (True if last day of month)
            - 'is_quarter_start': Boolean (True if first day of quarter)
            - 'is_quarter_end': Boolean (True if last day of quarter)
            - 'is_year_start': Boolean (True if first day of year)
            - 'is_year_end': Boolean (True if last day of year)
        inplace : bool, default False
            If True, modify DataFrame in place
        
        Returns
        ---
        pd.DataFrame
            DataFrame with new datetime features if inplace=False, or modified DataFrame if inplace=True
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Convert column to datetime if not already
        datetime_series = pd.to_datetime(df[column], errors='coerce')
        
        # Check if conversion was successful
        if datetime_series.isna().all():
            raise ValueError(f"Column '{column}' could not be converted to datetime")
        
        # Default features if not specified
        if features is None:
            features = ['year', 'month', 'day', 'hour', 'minute', 'second',
                       'day_of_week', 'day_of_year', 'week', 'quarter',
                       'is_weekend', 'is_weekday']
        
        # Validate features
        valid_features = ['year', 'month', 'day', 'hour', 'minute', 'second',
                          'day_of_week', 'day_of_year', 'week', 'quarter',
                          'is_weekend', 'is_weekday', 'is_month_start',
                          'is_month_end', 'is_quarter_start', 'is_quarter_end',
                          'is_year_start', 'is_year_end']
        
        invalid_features = [f for f in features if f not in valid_features]
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}. "
                           f"Valid features: {valid_features}")
        
        result_df = df.copy() if not inplace else df
        
        # Extract features
        feature_dict = {}
        if 'year' in features:
            feature_dict[f'{column}_year'] = datetime_series.dt.year
        if 'month' in features:
            feature_dict[f'{column}_month'] = datetime_series.dt.month
        if 'day' in features:
            feature_dict[f'{column}_day'] = datetime_series.dt.day
        if 'hour' in features:
            feature_dict[f'{column}_hour'] = datetime_series.dt.hour
        if 'minute' in features:
            feature_dict[f'{column}_minute'] = datetime_series.dt.minute
        if 'second' in features:
            feature_dict[f'{column}_second'] = datetime_series.dt.second
        if 'day_of_week' in features:
            feature_dict[f'{column}_day_of_week'] = datetime_series.dt.dayofweek
        if 'day_of_year' in features:
            feature_dict[f'{column}_day_of_year'] = datetime_series.dt.dayofyear
        if 'week' in features:
            feature_dict[f'{column}_week'] = datetime_series.dt.isocalendar().week
        if 'quarter' in features:
            feature_dict[f'{column}_quarter'] = datetime_series.dt.quarter
        if 'is_weekend' in features:
            feature_dict[f'{column}_is_weekend'] = datetime_series.dt.dayofweek.isin([5, 6])
        if 'is_weekday' in features:
            feature_dict[f'{column}_is_weekday'] = datetime_series.dt.dayofweek.isin([0, 1, 2, 3, 4])
        if 'is_month_start' in features:
            feature_dict[f'{column}_is_month_start'] = datetime_series.dt.is_month_start
        if 'is_month_end' in features:
            feature_dict[f'{column}_is_month_end'] = datetime_series.dt.is_month_end
        if 'is_quarter_start' in features:
            feature_dict[f'{column}_is_quarter_start'] = datetime_series.dt.is_quarter_start
        if 'is_quarter_end' in features:
            feature_dict[f'{column}_is_quarter_end'] = datetime_series.dt.is_quarter_end
        if 'is_year_start' in features:
            feature_dict[f'{column}_is_year_start'] = datetime_series.dt.is_year_start
        if 'is_year_end' in features:
            feature_dict[f'{column}_is_year_end'] = datetime_series.dt.is_year_end
        
        # Add new features to DataFrame
        for feature_name, feature_values in feature_dict.items():
            result_df[feature_name] = feature_values
        
        return result_df

    @staticmethod
    def split_column(df: pd.DataFrame, column: str, 
                     max_columns: int = None, column_prefix: str = None, reverse_split : bool = False,
                     inplace: bool = False):
        """
        Split an object column into multiple columns based on non-alphanumeric delimiters
        (excluding whitespace - whitespace is preserved).
        Supports Vietnamese characters.
        
        Parameters
        ---
        df : pd.DataFrame
            DataFrame containing the column to split
        column : str
            Column name to split
        max_columns : int, optional
            Maximum number of columns to create. If None, uses the maximum number of parts found
        column_prefix : str, optional
            Prefix for new column names. If None, uses '{column}_part'
        inplace : bool, default False
            If True, modify DataFrame in place
        
        Returns
        ---
        pd.DataFrame
            DataFrame with split columns if inplace=False, or modified DataFrame if inplace=True
        """
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_object_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not of object type. Cannot split non-string columns.")
        
        # Get column data
        column_data = df[column].copy()
        split_pattern = r'[^\w\s]+'
        
        # Split each value with UNICODE flag to support Vietnamese
        split_data = column_data.apply(
            lambda x: re.split(split_pattern, str(x), flags=re.UNICODE) if pd.notna(x) else []
        )
        
        # Trim whitespace from split results but keep the parts
        # Reverse the order of parts after splitting
        if reverse_split:
            split_data = split_data.apply(
                lambda x: list(reversed([part.strip() for part in x if part.strip()]))
            )
        else:
            split_data = split_data.apply(
                lambda x: [part.strip() for part in x if part.strip()]
            )
        
        # Determine maximum number of columns needed
        if max_columns is None:
            max_parts = split_data.apply(len).max()
            if pd.isna(max_parts) or max_parts == 0:
                max_parts = 1
            max_columns = int(max_parts)
        else:
            max_columns = int(max_columns)
        
        # Set column prefix
        if column_prefix is None:
            column_prefix = f"{column}_part"
        
        # Create new columns
        result_df = df.copy() if not inplace else df
        
        # Create columns for each part
        for i in range(max_columns):
            new_col_name = f"{column_prefix}_{i+1}"
            result_df[new_col_name] = split_data.apply(
                lambda x: x[i] if i < len(x) else None
            )
            result_df[new_col_name] = result_df[new_col_name].replace([None], np.nan)
        
        return result_df 

    @staticmethod
    def save_to_file(df: pd.DataFrame, file_path: str, 
                      file_type: str = None, encoding: str = "utf-8", **kwargs):
        """
        Internal method to save DataFrame to file.
        
        Parameters
        ---
        df : pd.DataFrame
            DataFrame to save
        file_path : str
            Path to save the file
        file_type : str, optional
            File type. Auto-detected from extension if None
        encoding : str, default 'utf-8'
            File encoding
        **kwargs : dict
            Additional parameters for pandas write functions
        """
        # Resolve relative paths from PROJECT root
        file_path = Path(file_path)
        
        # If path is relative, resolve it from PROJECT root
        if not file_path.is_absolute():
            # Get PROJECT root: from src/datapreprocessor/data_preprocessing.py go up 2 levels
            current_file_dir = Path(__file__).parent  # src/datapreprocessor/
            project_root = current_file_dir.parent.parent  # PROJECT/
            file_path = project_root / file_path
        
        file_path = file_path.resolve()
        
        # Detect file type from extension if not provided
        if file_type is None:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_type = "csv"
            elif suffix in [".xlsx", ".xls"]:
                file_type = suffix[1:]
            elif suffix == ".json":
                file_type = "json"
            elif suffix == ".parquet":
                file_type = "parquet"
            else:
                raise ValueError(f"Cannot auto-detect file type from extension: {suffix}")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_type == "csv":
                df.to_csv(file_path, encoding=encoding, index=False, **kwargs)
            elif file_type in ["xlsx", "xls"]:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == "json":
                orient = kwargs.pop('orient', 'records')
                df.to_json(file_path, orient=orient, force_ascii=False, **kwargs)
            elif file_type == "parquet":
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise RuntimeError(f"Error saving file {file_path}: {str(e)}") from e