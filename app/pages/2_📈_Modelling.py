import streamlit as st
import pandas as pd
from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset, DataPlotter


def write_helper_text(text: str) -> None:
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_dataset(datasets: list[Dataset]) -> pd.DataFrame:
    dataset_map = {dataset.name: dataset for dataset in datasets}
    data_name = st.selectbox('Select the dataset you want to use',
                             list(dataset_map.keys()), placeholder="Choose a dataset")
    selected_dataset = dataset_map[data_name]

    st.write(f"You selected the dataset: {selected_dataset.name}")
    dataframe = pd.read_csv(BytesIO(selected_dataset.data))
    st.write(dataframe.head())
    return dataframe


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
    dataframe = select_dataset(datasets)

    to_do = st.selectbox("What do you want to do with your data?", ["Create a plot", "Train a model"], placeholder="Choose an option")

    if to_do == 'Create a plot':
        plot(dataframe)
    elif to_do == 'Train a model':
        pass
        # Detect the features and generate a selection menu for selecting the input features (many) 
        # and one target feature. Based on the feature selections, prompt the user with 
        # the detected task type (i.e., classification or regression).

        # Prompt the user to select a model based on the task type.

        # Prompt the user to select a dataset split.

        # Prompt the user to select a set of compatible metrics.

        # Prompt the user with a beautifully formatted pipeline summary with all the configurations.

        # Train the class and report the results of the pipeline.

        # Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.
