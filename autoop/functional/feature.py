from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

    
def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    
    data_frame = dataset.read().dropna()
    
    if data_frame.isnull().values.any():
        raise ValueError("The dataset currently contains NaN values, which is not allowed.")

    for col in data_frame.columns:
        col_dtype = data_frame[col].dtype
        if pd.api.types.is_numeric_dtype(data_frame[col]):
            feature_type = "numerical"
        elif isinstance(col_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data_frame[col]):
            feature_type = "categorical"
        else:
            raise ValueError(f"Column '{col}' contains an unsupported feature type.")

        feature = Feature(name=col, type=feature_type)
        features.append(feature)
    
    return features

