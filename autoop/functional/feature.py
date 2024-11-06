
from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from pydantic import BaseModel, Field


class Feature(BaseModel):
    name: str = Field(
        ..., description="The feature's name")
    data_type: str = Field(
        ..., description="The feature's data type (e.g., 'numerical', 'categorical')")
    is_categorical: bool = Field(
        default=False, description="Whether the feature is categorical")
    is_continuous: bool = Field(
        default=False, description="Whether the feature is continuous")
    

    def encode(self):
        """Encode categorical features into numerical values."""
        if self.is_categorical:
            pass

    def normalize(self):
        """Normalize continuous features."""
        if self.is_continuous:
            pass

    def __repr__(self):
        return f"Feature(name={self.name}, data_type={self.data_type}, 
        is_categorical={self.is_categorical}, is_continuous={self.is_continuous})"
    
def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    
    data_frame = dataset.read()
    
    if data_frame.isnull().values.any():
        raise ValueError("The dataset currently contains NaN values, which is not allowed.")

    for col in data_frame.columns:
        if pd.api.types.is_numeric_dtype(data_frame[col]):
            feature_type = "numerical"
        elif pd.api.types.is_categorical_dtype(data_frame[col]) or pd.api.types.is_object_dtype(data_frame[col]):
            feature_type = "categorical"
        else:
            raise ValueError(f"Column '{col}' contains an unsupported feature type.")

        feature = Feature(name=col, type=feature_type)
        features.append(feature)
    
    return features

