
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name: str = Field(
        ..., description="The feature's name")
    type: str = Field(
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
        return (f"Feature(name={self.name}, data_type={self.data_type}, "
            f"is_categorical={self.is_categorical}, is_continuous={self.is_continuous})")
