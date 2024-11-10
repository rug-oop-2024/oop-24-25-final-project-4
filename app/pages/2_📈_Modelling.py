import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset, DataPlotter
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification.decision_tree_classifier import DecisionTreeClassifierModel
from autoop.core.ml.model.classification.k_nearest_neighbour import KNNClassifier
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifierModel
from autoop.core.ml.model.regression.decision_tree_regressor import DecisionTreeRegressorModel
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.random_forest_regressor import RandomForestRegressorModel
from autoop.core.ml.metric import MeanSquaredError, Accuracy, Precision, Recall, MeanAbsoluteError, R2Score
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from typing import Any


def write_helper_text(text: str) -> None:
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_dataset(datasets: list[Dataset]) -> tuple[Dataset, pd.DataFrame]:
    dataset_map = {dataset.name: dataset for dataset in datasets}
    data_name = st.selectbox('Select the dataset you want to use',
                             list(dataset_map.keys()), placeholder="Choose a dataset")
    selected_dataset = dataset_map[data_name]

    st.write(f"You selected the dataset: {selected_dataset.name}")
    dataframe = pd.read_csv(BytesIO(selected_dataset.data))
    st.write(dataframe.head())
    return selected_dataset, dataframe


def plot(data: pd.DataFrame) -> None:
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
            st.error("""One of the variables you selected is a catagorical value. 
                     It is not possible to plot this in a 3d plot.
                        Please select a non-categorical variable""")
    if len(columns) > 3:
        st.error("Please select at most 3 columns to plot.")
    if fig:
        st.pyplot(fig)


def pipeline_summary(name: str, input_features: list[str],
                    target_feature: str, task_type: str,
                    model: str, split: int, metrics: list[str]) -> None:
    st.subheader("Pipeline Summary")
    st.write(f"**Dataset**: {name}")
    st.write(f"**Input features**: {', '.join(input_features)}")
    st.write(f"**Target feature**: {target_feature}")
    st.write(f"**Task type**: {task_type}")
    st.write(f"**Model**: {model}")
    st.write(f"""**Data split**: {round(split * 100)}% for training,
             {round(100 - split * 100)}% for testing""")
    st.write(f"**Metrics**: {', '.join(metrics)}")


def display_pipeline_results(results: dict[str, Any]):
    train_metrics = {metric.__class__.__name__: value for metric, value in results["train_metrics"]}
    test_metrics = {metric.__class__.__name__: value for metric, value in results["test_metrics"]}
    train_predictions = results["train_predictions"]
    test_predictions = results["test_predictions"]

    st.subheader("Training Metrics")
    train_metrics_df = pd.DataFrame(list(train_metrics.items()), columns=["Metric", "Value"])
    st.write(train_metrics_df)

    st.subheader("Testing Metrics")
    test_metrics_df = pd.DataFrame(list(test_metrics.items()), columns=["Metric", "Value"])
    st.write(test_metrics_df)

    train_predictions_sample = train_predictions[:10]
    test_predictions_sample = test_predictions[:10]

    predictions_df = pd.DataFrame({
        "Train Predictions": train_predictions_sample,
        "Test Predictions": test_predictions_sample
        })

    st.subheader("Sample Predictions")
    st.write("Showing the first 10 predictions for both training and testing sets:")
    st.write(predictions_df)

def create_pipeline(data: pd.DataFrame) -> None:
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

    features = detect_feature_types(data)
    feature_map = {feature.name: feature for feature in features}

    input_feature_names = st.multiselect('Select input features',list(feature_map.keys()), placeholder="Choose a dataset")
    input_features = [feature_map[input_feature_name] for input_feature_name in input_feature_names]
    target_feature_name = st.selectbox("Select a target feature", list(feature_map.keys()))
    target_feature = feature_map[target_feature_name]

    if target_feature_name in input_feature_names:
        st.error("Target feature can not be an input feature, please select a different target or input feature.")
    elif input_feature_names:
        if target_feature.type == "categorical":
            detected_task_type = "Classification"
        elif target_feature.type == "numerical":
            detected_task_type = "Regression"
        st.write(
            f"The following task type was detected: {detected_task_type}"
            )

        if detected_task_type == "Regression":
            selected_model = st.selectbox("Select a model", list(model_map.keys())[-3:])
        else:
            selected_model = st.selectbox("Select a model", list(model_map.keys())[:3])
        ModelClass = model_map[selected_model]
        model = ModelClass()

        dataset_split = st.slider("Split your dataset", 0.0, 1.0, 0.8)
        st.write(f"Data used for training: {round(dataset_split * 100)}%")
        st.write(f"Data used for testing: {round(100 - dataset_split * 100)}%")

        if detected_task_type == "Regression":
            selected_metrics = st.multiselect("Select the metrics you want to use",
                                              list(metric_map.keys())[-3:])
        else:
            selected_metrics = st.multiselect("Select the metrics you want to use",
                                              list(metric_map.keys())[:3])
        metrics_list = [metric_map[metric] for metric in selected_metrics]

    if input_feature_names and selected_model and selected_metrics:
        pipeline_summary(selected_dataset.name, input_feature_names,
                         target_feature_name, detected_task_type,
                         selected_model, dataset_split, selected_metrics)

        pipeline_button = st.button("Train the model")
        pipeline_trained = False
        if pipeline_button:
            pipeline = Pipeline(metrics_list, selected_dataset, model, input_features, target_feature, dataset_split)
            results = pipeline.execute()
            pipeline_trained = True
        if pipeline_trained:
            display_pipeline_results(results)
            

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

    to_do = st.selectbox("What do you want to do with your data?", ["Train a model", "Create a plot"], placeholder="Choose an option")

    if to_do == 'Create a plot':
        plot(dataframe)
    elif to_do == 'Train a model':
        create_pipeline(dataframe)

        # Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.
