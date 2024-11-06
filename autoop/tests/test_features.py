import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd
import numpy as np
import pytest

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_detect_features_continuous(self):
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")
        
    def test_detect_features_with_categories(self):
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns, features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(lambda x: x.name in categorical_columns, features):
            self.assertEqual(detected_feature.type, "categorical")

def test_detect_features_with_nan_values(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, np.nan, 6],
        })
        dataset = Dataset.from_dataframe(
            name="test_with_nan",
            asset_path="test_with_nan.csv",
            data=df,
        )
        dataset = Dataset.from_dataframe(
            name="test_with_nan",
            asset_path="test_with_nan.csv",
            data=df,
            metadata={"description": "Dataset with NaN values for testing"},
            tags=["nan", "test"]
        )
        # Assert that ValueError is raised for NaN values
        with self.assertRaises(ValueError) as context:
            detect_feature_types(dataset)
        self.assertEqual(str(context.exception), "The dataset currently contains NaN values, which is not allowed.")

def test_detect_features_with_unsupported_type(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 2, 3], 
        })
        dataset = Dataset.from_dataframe(
            name="test_with_mixed_types",
            asset_path="test_with_mixed_types.csv",
            data=df,
        )
        dataset = Dataset.from_dataframe(
            name="test_with_mixed_types",
            asset_path="test_with_mixed_types.csv",
            data=df,
            metadata={"description": "Dataset with mixed types for testing"},
            tags=["mixed", "test"]
        )
        with self.assertRaises(ValueError) as context:
            detect_feature_types(dataset)
        self.assertIn("Column 'col2' contains an unsupported feature type.", str(context.exception))
