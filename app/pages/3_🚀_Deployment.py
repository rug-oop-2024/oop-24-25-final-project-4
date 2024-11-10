import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact


def select_pipeline(pipelines: list[Artifact]) -> Artifact:
    pipeline_map = {pipeline.name: pipeline for pipeline in pipelines}
    pipeline_name = st.selectbox(
        'Select the dataset you want to use',
        list(pipeline_map.keys()),
        placeholder="Choose a dataset"
    )
    selected_pipeline = pipeline_map[pipeline_name]

    st.write(f"You selected the pipeline: {selected_pipeline.name}")

    return selected_pipeline


def show_summary(pipeline: Artifact) -> None:
    pass


st.set_page_config(page_title="Deployment", page_icon="🚀")
st.title("Deployment")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

if not pipelines:
    st.write("""You have no existing pipelines.
             Please navigate to the Modelling page
             if you wish to create a pipeline.""")
else:
    # Allow the user to select existing pipelines and based on the selection show a pipeline summary.
    pipeline = select_pipeline(pipelines)
    show_summary(pipeline)

    #  Once the user loads a pipeline, prompt them to provide a CSV on which they can perform predictions.
