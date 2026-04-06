"""Single Cell — Settings tab: import, deploy and manage models."""

import streamlit as st
import time
from genesis_workbench.models import ModelCategory
from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog)


def _reset_available_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["available_single_cell_models_df"]
        st.rerun()


def render(available_models_df, deployed_models_df):
    p1, p2 = st.columns([2, 1])

    with p1:
        st.markdown("###### Import Models:")
        with st.form("import_model_form"):
            col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
            with col1:
                import_model_source = st.selectbox("Source:", ["Unity Catalog", "Hugging Face", "PyPi"], label_visibility="visible")
            with col2:
                import_button = st.form_submit_button('Import')

        if import_button:
            if import_model_source == "Unity Catalog":
                display_import_model_uc_dialog(ModelCategory.SINGLE_CELL, success_callback=_reset_available_models)

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
                             "Endpoint Name": None
                         })
    else:
        st.write("There are no deployed models")
