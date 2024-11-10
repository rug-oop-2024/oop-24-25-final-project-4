import streamlit as st
import pandas as pd
from io import BytesIO
import re
import os
import pickle
import numpy as np

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset, DataPlotter
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeClassifierModel)
from autoop.core.ml.model.classification.k_nearest_neighbour import (
    KNNClassifier)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForestClassifierModel)
from autoop.core.ml.model.regression.decision_tree_regressor import (
    DecisionTreeRegressorModel)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.random_forest_regressor import (
    RandomForestRegressorModel)
from autoop.core.ml.metric import (
    MeanSquaredError,
    Accuracy, Precision,
    Recall,
    MeanAbsoluteError,
    R2Score)
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
from typing import Any

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}

if 'pipeline_trained' not in st.session_state:
    st.session_state.pipeline_trained = False

if 'pipeline_name' not in st.session_state:
    st.session_state.pipeline_name = ''

if 'pipeline_version' not in st.session_state:
    st.session_state.pipeline_version = ''


def write_helper_text(text: str) -> None:
    """
    Display helper text in the Streamlit app.

    Args:
        text (str): The helper text to display.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_dataset(datasets: list[Dataset]) -> tuple[Dataset, pd.DataFrame]:
    """
    Allows the user to select a dataset from the available datasets.

    Args:
        datasets (list[Dataset]): A list of available Dataset objects.

    Returns:
        tuple[Dataset, pd.DataFrame]: The selected dataset
        and its corresponding DataFrame.
    """
    dataset_map = {dataset.name: dataset for dataset in datasets}
    data_name = st.selectbox('Select the dataset you want to use',
                             list(dataset_map.keys()),
                             placeholder="Choose a dataset")
    selected_dataset = dataset_map[data_name]

    st.write(f"You selected the dataset: {selected_dataset.name}")
    dataframe = pd.read_csv(BytesIO(selected_dataset.data))
    st.write(dataframe.head())
    return selected_dataset, dataframe


def plot(data: pd.DataFrame) -> None:
    """
    Allows the user to plot selected columns of the dataset.

    Args:
        data (pd.DataFrame): The dataset to plot.

    This function provides options to plot histograms,
    scatter plots, and 3D plots.
    """
    plotter = DataPlotter(data)
    columns = st.multiselect("Select columns to plot", data.columns)
    fig = None
    if len(columns) == 0:
        st.write("Please select at least one column to plot.")
    if len(columns) == 1:
        fig = plotter.hist_1d(columns[0])
    if len(columns) == 2:
        fig = plotter.scatter_2d(columns[0], columns[1])
    if len(columns) == 3:
        try:
            fig = plotter.scatter_3d(columns[0], columns[1], columns[2])
        except ValueError:
            st.error("""One of the variables you
                     selected is a catagorical value.
                     It is not possible to plot this in a 3d plot.
                        Please select a non-categorical variable""")
    if len(columns) > 3:
        st.error("Please select at most 3 columns to plot.")
    if fig:
        st.pyplot(fig)


def pipeline_summary(name: str, input_features: list[str],
                     target_feature: str, task_type: str,
                     model: str, split: int, metrics: list[str]) -> None:
    """
    Display a summary of the pipeline configuration.

    Args:
        name (str): The name of the dataset.
        input_features (list[str]): The list of input features.
        target_feature (str): The target feature.
        task_type (str): The type of the task
        ("Classification" or "Regression").
        model (str): The selected model.
        split (int): The dataset split ratio for training and testing.
        metrics (list[str]): The list of selected metrics.
    """
    st.subheader("Pipeline Summary")
    st.write(f"**Dataset**: {name}")
    st.write(f"**Input features**: {', '.join(input_features)}")
    st.write(f"**Target feature**: {target_feature}")
    st.write(f"**Task type**: {task_type}")
    st.write(f"**Model**: {model}")
    st.write(f"""**Data split**: {round(split * 100)}% for training,
             {round(100 - split * 100)}% for testing""")
    st.write(f"**Metrics**: {', '.join(metrics)}")


def display_pipeline_results(results: dict[str, Any]) -> None:
    """
    Display the results of the trained pipeline
    including metrics and predictions.

    Args:
        results (dict[str, Any]): The results of the pipeline execution.
    """
    train_metrics = {
        metric.__class__.__name__: value
        for metric, value in results["train_metrics"]
    }
    test_metrics = {
        metric.__class__.__name__: value
        for metric, value in results["test_metrics"]
    }
    train_predictions = results["train_predictions"]
    test_predictions = results["test_predictions"]

    st.subheader("Training Metrics")
    train_metrics_df = pd.DataFrame(
        list(train_metrics.items()),
        columns=["Metric", "Value"]
    )
    st.write(train_metrics_df)

    st.subheader("Testing Metrics")
    test_metrics_df = pd.DataFrame(
        list(test_metrics.items()),
        columns=["Metric", "Value"]
    )
    st.write(test_metrics_df)

    train_predictions_sample = train_predictions[:10]
    test_predictions_sample = test_predictions[:10]

    if isinstance(train_predictions_sample, (list, np.ndarray)) and (
        isinstance(train_predictions_sample[0], (list, np.ndarray))):
        train_predictions_sample = [
            item[0] for item in train_predictions_sample]
    if isinstance(test_predictions_sample, (list, np.ndarray)) and (
        isinstance(test_predictions_sample[0], (list, np.ndarray))):
        test_predictions_sample = [
            item[0] for item in test_predictions_sample]

    predictions_df = pd.DataFrame({
        "Train Predictions": train_predictions_sample,
        "Test Predictions": test_predictions_sample
    })

    st.subheader("Sample Predictions")
    st.write("""Showing the first 10 predictions
             for both training and testing sets:""")
    st.write(predictions_df)


def create_pipeline(data: pd.DataFrame) -> tuple[dict[str, Any], Model]:
    """
    Create and configure a machine learning pipeline based on user input.

    Args:
        data (pd.DataFrame): The dataset to use for creating the pipeline.

    Returns:
        dict[str, Any]: The results from the pipeline execution.
    """
    model_map = {"Decision Tree Classifier": DecisionTreeClassifierModel,
                 "K Nearest Neighbours": KNNClassifier,
                 "Random Forest Classifier": RandomForestClassifierModel,
                 "Decision Tree Regressor": DecisionTreeRegressorModel,
                 "Multiple Linear Regression": MultipleLinearRegression,
                 "Random Forest Regressor": RandomForestRegressorModel}

    metric_map = {"Accuracy": Accuracy,
                  "Precision": Precision,
                  "Recall": Recall,
                  "Mean Squared Error": MeanSquaredError,
                  "Mean Absolute Error": MeanAbsoluteError,
                  "R^2 Score": R2Score}

    model = None

    features = detect_feature_types(data)
    feature_map = {feature.name: feature for feature in features}

    input_feature_names = st.multiselect(
        'Select input features',
        list(feature_map.keys()),
        placeholder="Choose a dataset"
    )
    input_features = [
        feature_map[input_feature_name]
        for input_feature_name in input_feature_names
    ]
    target_feature_name = st.selectbox(
        "Select a target feature",
        list(feature_map.keys())
    )
    target_feature = feature_map[target_feature_name]

    if target_feature_name in input_feature_names:
        st.error("""Target feature can not be an input feature,
                 please select a different target or input feature.""")
    elif input_feature_names:
        if target_feature.type == "categorical":
            detected_task_type = "Classification"
        elif target_feature.type == "numerical":
            detected_task_type = "Regression"
        st.write(
            f"The following task type was detected: {detected_task_type}"
        )

        if detected_task_type == "Regression":
            selected_model = st.selectbox(
                "Select a model",
                list(model_map.keys())[-3:]
            )
        else:
            selected_model = st.selectbox(
                "Select a model",
                list(model_map.keys())[:3]
            )
        ModelClass = model_map[selected_model]
        model = ModelClass()

        dataset_split = st.slider("Split your dataset", 0.0, 1.0, 0.8)
        st.write(f"Data used for training: {round(dataset_split * 100)}%")
        st.write(f"Data used for testing: {round(100 - dataset_split * 100)}%")

        if detected_task_type == "Regression":
            selected_metrics = st.multiselect(
                "Select the metrics you want to use",
                list(metric_map.keys())[-3:]
            )
        else:
            selected_metrics = st.multiselect(
                "Select the metrics you want to use",
                list(metric_map.keys())[:3]
            )
        metrics_list = [metric_map[metric] for metric in selected_metrics]

    results = {}
    if input_feature_names and selected_model and selected_metrics:
        pipeline_summary(selected_dataset.name, input_feature_names,
                         target_feature_name, detected_task_type,
                         selected_model, dataset_split, selected_metrics)

        pipeline_button = st.button("Train the model")
        if pipeline_button:
            pipeline = Pipeline(
                metrics_list,
                selected_dataset, model,
                input_features,
                target_feature,
                dataset_split
            )
            results = pipeline.execute()
            st.session_state.pipeline_results = results
            st.session_state.pipeline_trained = True
            if st.session_state.pipeline_trained:
                display_pipeline_results(st.session_state.pipeline_results)

    return st.session_state.get("pipeline_results", {}), model


def save_pipeline(name: str, version: str,
                  model: Model, results: dict) -> None:
    """
    Save the machine learning pipeline,
    including the trained model and metadata,
    as an Artifact in the AutoML system.

    Args:
        name (str): The name of the pipeline.
        version (str): The version of the pipeline, e.g., '1.0.0'.
        model: The trained model object
        (e.g., scikit-learn, XGBoost model).
        results (dict): Dictionary containing pipeline
        metadata such as metrics, input features, target feature,
        split ratios, and artifacts.

    Saves:
        An Artifact containing the serialized model
        and metadata in the AutoML registry.
    """
    pipeline_dir = "./pipelines"
    os.makedirs(pipeline_dir, exist_ok=True)
    asset_path = os.path.join(pipeline_dir, f"{name}_{version}.pkl")

    metadata = {
        "name": name,
        "version": version,
        "input_features": results.get("input_features", []),
        "target_feature": results.get("target_feature", ""),
        "split": results.get("split", ""),
        "metrics": results.get("metrics", {}),
    }

    artifact_data = {
        "model": model,
        "metadata": metadata,
    }

    serialized_data = pickle.dumps(artifact_data)

    with open(asset_path, "wb") as f:
        f.write(serialized_data)

    artifact = Artifact(
        name=f"{name}_{version}",
        version=version,
        data=serialized_data,
        asset_path=asset_path,
        type="pipeline",
        tags=["machine_learning", "pipeline", name],
        metadata=metadata
    )

    automl = AutoMLSystem.get_instance()
    automl.registry.register(artifact)

    st.success(f"""Pipeline '{name}' has been uploaded
               and registered successfully. You can now use it
               on the deployment page""")


st.set_page_config(page_title="Modelling", page_icon="📈")
st.title("Modelling")
write_helper_text("""In this section, you can design a machine learning
                  pipeline to train a model on a dataset.""")

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

if not datasets:
    st.write("""You currently don't have any datasets uploaded.
             Please upload your datasets on the Dataset page.""")
else:
    selected_dataset, dataframe = select_dataset(datasets)

    to_do = st.selectbox(
        "What do you want to do with your data?",
        ["Train a model", "Create a plot"],
        placeholder="Choose an option"
    )

    if to_do == 'Create a plot':
        plot(dataframe)
    elif to_do == 'Train a model':
        pipeline_results, model = create_pipeline(dataframe)
        if pipeline_results:
            name = st.text_input(
                "Enter a name for the pipeline:",
                placeholder="Pipeline name",
                value=st.session_state.get('pipeline_name', ''))
            version = st.text_input(
                "Enter a version for the pipeline:",
                placeholder="1.0.0",
                value=st.session_state.get('pipeline_version', ''))

            st.session_state.pipeline_name = name
            st.session_state.pipeline_version = version

            pattern = r'^\d+(\.\d+)*$'
            if name and version and not re.match(pattern, version):
                st.error("Please enter a valid number for version")
            if st.session_state.pipeline_name and (
                st.session_state.pipeline_version and (
                    st.button("Save Pipeline"))):
                save_pipeline(st.session_state.pipeline_name,
                              st.session_state.pipeline_version,
                              model,
                              pipeline_results)
