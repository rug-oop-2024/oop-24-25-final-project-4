import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact

def select_pipeline(pipelines: list[Artifact]):
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

def show_summary(pipeline):
    st.subheader("Pipeline Summary")
    st.write("Pipeline Name:", pipeline.name)
    st.write("Pipeline Version:", pipeline.version)

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

    uploaded_file = st.file_uploader(
        "Upload a CSV file for prediction",
        type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", input_data.head())

        predictions = selected_pipeline.object.predict(input_data)
        st.write("Predictions:", predictions)
        
        prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
        st.download_button(
            label="Download Predictions as CSV",
            data=prediction_df.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )
