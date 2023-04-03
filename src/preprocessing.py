import numpy as np
import pandas as pd

def apply_clamping(df: pd.DataFrame, limit: float = 10, reduction_quantile: float = .95) -> pd.DataFrame:
    """
    Applies clamping on the provided dataframe.
    Requires dataframe to consist of numeric features.
    If the maximum of a feature column is greater than *limit* times the median of the column, then we clamp the feature column to the quantile specified by *reduction_quantile*
    """
    for feature in df.columns:
        if df[feature].max() > df[feature].median() * limit and df[feature].max() > limit:
            df[feature] = np.where(
                df[feature] < df[feature].quantile(reduction_quantile),
                df[feature], 
                df[feature].quantile(reduction_quantile))

    return df


def reduce_top_n(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """
    To avoid the curse of dimensionality, we reduce categories of categorical features in provided dataframe to the top *n* features.
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for feature in cat_cols:
        if df[feature].nunique() > n:
            top_n = df[feature].value_counts().head(n).index
            df[feature] = np.where(
                df[feature].isin(top_n), 
                df[feature], 
                '-')
    
    return df


def feature_selection_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = reduce_top_n(df, n)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df = pd.concat([df[num_cols], pd.get_dummies(df[cat_cols])], axis=1)

    df = apply_clamping(df)
    