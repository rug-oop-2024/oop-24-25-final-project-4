import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def save_dataset(file: Dataset, name: str) -> None:
    """
    Saves the uploaded dataset to a specified location and registers it
    with the AutoML system.

    Args:
        file (Dataset): The uploaded dataset file.
        name (str): The name to assign to the dataset.

    Raises:
        UnicodeDecodeError: If there is an encoding issue while reading the CSV file.
    """
    data = pd.read_csv(file)

    asset_path = f"datasets/{name}.csv"

    new_dataset = Dataset.from_dataframe(
        data,
        name=name,
        asset_path=asset_path,
        version="1.0.0",
    )

    try:
        automl.registry.register(new_dataset)
        st.success(
            f"""Dataset '{dataset_name}' has been uploaded
                   and registered successfully.
                   Refresh the page to see it under 'Availabe Datasets'"""
        )
    except UnicodeDecodeError:
        st.error(
            """Error: Dataset could not be
                 registered due to encoding issues."""
        )


def display_datasets(datasets: list[Dataset]) -> None:
    """
    Displays the list of available datasets to the user.

    Args:
        datasets (list[Dataset]): A list of Dataset objects to display.

    This function iterates through the datasets and displays their name and version.
    """
    number_datasets = len(datasets)
    if number_datasets == 1:
        st.write("You have 1 dataset saved:")
    else:
        st.write(f"You have {number_datasets} datasets saved:")
    for dataset in datasets:
        dataset_name = dataset.name
        st.write(f"- Name: {dataset_name}, Version: {dataset.version}")


st.set_page_config(page_title="Datasets", page_icon="💾")
st.title("Dataset Management")

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")


st.subheader("Upload a New Dataset")
# Let the user upload a dataset and give it a name
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
dataset_name = st.text_input("Dataset Name", placeholder="Enter dataset name")

upload_button = st.button("Upload Dataset")

# Save the dataset
if upload_button and uploaded_file and dataset_name:
    save_dataset(uploaded_file, dataset_name)

st.subheader("Available Datasets")

if datasets:
    display_datasets(datasets)
else:
    st.write(
        """You currently have no datasets uploaded.
             Please upload a dataset at the top of the page."""
    )
