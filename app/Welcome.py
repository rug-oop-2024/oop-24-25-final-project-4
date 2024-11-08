from autoop.core.ml.artifact import Artifact
import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="🏠",
)
st.sidebar.success("Select a page above.")
st.markdown(open("app/core/welcome.md", encoding="utf-8").read())