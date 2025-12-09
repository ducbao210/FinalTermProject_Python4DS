from pandas.api.types import is_numeric_dtype


def ensure_columns_exist(df, cols):
    """
    Validate that all specified columns exist in the DataFrame.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    cols : list
        List of column names to check.

    Raises
    ---
    ValueError
        If any column in `cols` is missing from the DataFrame.
    """
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in dataframe.")


def ensure_numeric(df, cols):
    """
    Validate that specified columns contain numeric data.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    cols : list
        Columns to validate.

    Raises
    ---
    TypeError
        If any column in `cols` is not numeric.
    """
    for c in cols:
        if not is_numeric_dtype(df[c]):
            raise TypeError(f"Column '{c}' must be numeric.")


def ensure_categorical_or_low_cardinality(df, cols, max_unique=20):
    """
    Validate that numeric columns have low cardinality suitable for categorical plots.

    Parameters
    ---
    df : pd.DataFrame
        Input DataFrame.
    cols : list
        Columns to validate.
    max_unique : int, optional
        Maximum allowed unique values for numeric columns. Defaults to 20.

    Raises
    ---
    TypeError
        If any numeric column has more unique values than `max_unique`.
    """
    for c in cols:
        if is_numeric_dtype(df[c]) and df[c].nunique() > max_unique:
            raise TypeError(f"Column '{c}' is too high-cardinality for this plot.")


def to_list(x):
    """
    Convert input to a list if it is not already a list.

    Parameters
    ---
    x : any
        Input value or iterable.

    Returns
    ---
    list
        List-wrapped value.
    """
    return x if isinstance(x, list) else [x]
