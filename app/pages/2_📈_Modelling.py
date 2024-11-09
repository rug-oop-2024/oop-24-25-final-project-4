import streamlit as st
import pandas as pd
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
        features = detect_feature_types(dataframe)
        feature_map = {feature.name: feature for feature in features}
        #feature_column_map = {feature.name: column for feature, column in zip(features, dataframe.columns)}
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
            model = model_map[selected_model]
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
            st.subheader("Pipeline Summary")
            st.write(f"**Dataset**: {selected_dataset.name}")
            st.write(f"**Input features**: {', '.join(input_feature_names)}")
            st.write(f"**Target feature**: {target_feature_name}")
            st.write(f"**Task type**: {detected_task_type}")
            st.write(f"**Model**: {selected_model}")
            st.write(f"""**Data split**: {round(dataset_split * 100)}% for training,
                     {round(100 - dataset_split * 100)}% for testing""")
            st.write(f"**Metrics**: {', '.join(selected_metrics)}")
            pipeline_button = st.button("Train the model")
            if pipeline_button:
                pipeline = Pipeline(metrics_list, selected_dataset, model, input_features, target_feature, dataset_split)
                results = pipeline.execute()
                st.write(f"train_metrics: {results['train_metrics']}")
                st.write(f"train_predictions: {results['train_predictions']}")
                st.write(f"test_metrics: {results['test_metrics']}")
                st.write(f"test_predictions: {results['test_predictions']}")


        # Train the class and report the results of the pipeline.

        # Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.
