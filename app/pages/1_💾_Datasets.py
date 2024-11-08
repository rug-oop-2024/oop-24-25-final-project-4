import streamlit as st
import pandas as pd
import json

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Datasets", page_icon="💾")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")
st.subheader("Available Datasets")
datasets = automl.registry.list(type="dataset")
if datasets:
    number_datasets = len(datasets)
    if number_datasets == 1:
        st.write("You have 1 dataset:")
    else:
        st.write(f"You have {number_datasets} datasets:")
    for dataset in datasets:
        dataset_name = dataset.name
        st.write(f"- Name: {dataset_name}, Version: {dataset.version}")
else:
    st.write("No datasets available.")

# File uploader for new CSV files
st.subheader("Upload a New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
dataset_name = st.text_input("Dataset Name", placeholder="Enter dataset name")

if uploaded_file and dataset_name:
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)

    data_bytes = data.to_csv(index=False).encode('utf-8')

    # Define the asset path for saving
    asset_path = f"datasets/{dataset_name}.csv"

    # Step 2: Create a Dataset object using the encoded data
    new_dataset = Dataset.from_dataframe(
        data, 
        name=dataset_name, 
        asset_path=asset_path,
        version="1.0.0",
    )

    # Register the new Dataset in the artifact registry
    try:
        # Decode the data to check for encoding issues
        decoded_data = data_bytes.decode('utf-8')  # Only for validation, not saved

        # If decoding succeeds, register the dataset
        automl.registry.register(new_dataset)
        st.success(f"Dataset '{dataset_name}' has been uploaded and registered successfully.")
    except (UnicodeDecodeError) as e:
        # Handle encoding errors
        st.error("Error: Dataset could not be registered due to encoding issues.")
