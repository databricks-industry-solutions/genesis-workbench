"""Protein Studies — Settings tab: deploy and manage models."""

import streamlit as st
from utils.streamlit_helper import display_deploy_model_dialog


def render(available_models_df, deployed_models_df):
    st.markdown("###### Available Models:")
    with st.form("deploy_model_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_model_for_deploy = st.selectbox("Model:", available_models_df["model_labels"], label_visibility="collapsed")
        with col2:
            deploy_button = st.form_submit_button('Deploy')
    if deploy_button:
        display_deploy_model_dialog(selected_model_for_deploy)

    if len(deployed_models_df) > 0:
        with st.form("modify_deployed_model_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("###### Deployed Models")
            with col2:
                st.form_submit_button("Manage")

            st.dataframe(deployed_models_df,
                         use_container_width=True,
                         hide_index=True,
                         on_select="rerun",
                         selection_mode="single-row",
                         column_config={
                             "Model Id": None,
                             "Deploy Id": None,
                             "Endpoint Name": None,
                             "Model Name": None,
                             "Source Version": None,
                         })
    else:
        st.write("There are no deployed models")
