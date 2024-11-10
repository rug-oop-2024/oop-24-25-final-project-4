import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
import pickle
import numpy as np


def select_pipeline(pipelines: list[Artifact]) -> Artifact:
    """
    Allow the user to select a saved pipeline
    from the list of available pipelines.

    Args:
        pipelines (list): List of Artifact objects
        representing saved pipelines.

    Returns:
        Artifact: The selected pipeline.
    """
    pipeline_map = {
        f"""{pipeline.name} (v{pipeline.version})""":
        pipeline for pipeline in pipelines}
    pipeline_name = st.selectbox(
        "Select the pipeline you want to use:",
        list(pipeline_map.keys()))
    selected_pipeline = pipeline_map[pipeline_name]

    st.write(f"""You selected the pipeline:
             {selected_pipeline.name},
             version {selected_pipeline.version}""")
    return selected_pipeline


def show_summary(pipeline: Artifact) -> None:
    """
    Display a summary of the selected pipeline, including its name and version.

    Args:
        pipeline (Artifact): The selected pipeline.
    """
    st.subheader("Pipeline Summary")
    st.write("**Pipeline Name**:", pipeline.name)
    st.write("**Pipeline Version**:", pipeline.version)

    metadata = pipeline.metadata

    if metadata:
        st.write("**Input Features**:",
                 ", ".join(metadata.get(
                     "input_feature_names", [])))
        st.write("**Target Feature**:",
                 metadata.get("target_feature_name", "N/A"))
        st.write("**Task Type**:", metadata.get("task_type", "N/A"))
        st.write("**Model**:", metadata.get("model", "N/A"))
        split = metadata.get('dataset_split', None)
        if split is None or not isinstance(split, (int, float)):
            split = 0.8
        st.write(f"**Data Split**: {round(split * 100)}% for training, "
                 f"{round((1 - split) * 100)}% for testing")
        st.write("**Metrics**:", ", ".join(metadata.get("metrics_names", [])))


def display_pipeline_results(results: dict) -> None:
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


st.set_page_config(page_title="Deployment", page_icon="🚀")
st.title("Deployment")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

if not pipelines:
    st.write("""You have no existing pipelines.
             Please navigate to the Modeling page
             to create and save a pipeline.""")
else:
    selected_pipeline = select_pipeline(pipelines)
    show_summary(selected_pipeline)

    loaded_data = pickle.loads(selected_pipeline.data)

    model = loaded_data["model"]

    st.subheader("Pipeline Results")
    display_pipeline_results(loaded_data["results"])

    st.subheader("Upload a dataset for predicition")
    uploaded_file = st.file_uploader(
        "Upload a CSV file for prediction", type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", input_data.head())

        predictions = model.predict(input_data)
        st.write("Predictions:", predictions)

        prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
        st.download_button(
            label="Download Predictions as CSV",
            data=prediction_df.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )
