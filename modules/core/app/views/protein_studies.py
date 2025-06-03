import streamlit as st
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models,                                      
                                      get_gwb_model_info,
                                      deploy_model)

from utils.streamlit_helper import (get_app_context, 
                                    display_import_model_uc_dialog,
                                    display_deploy_model_dialog)


st.title(":material/biotech: Protein Structure Prediction")

def display_protein_studies_settings(available_models_df,deployed_models_df):
    p1,p2 = st.columns([2,1])

    with p1:

        st.markdown("###### Available Models:")
        with st.form("deploy_model_form"):
            col1, col2, = st.columns([1,1])    
            with col1:
                selected_model_for_deploy = st.selectbox("Model:",available_models_df["model_labels"],label_visibility="collapsed",)

            with col2:
                deploy_button = st.form_submit_button('Deploy')
        if deploy_button:
            display_deploy_model_dialog(selected_model_for_deploy)


    if len(deployed_models_df) > 0:
        with st.form("modify_deployed_model_form"):
            col1,col2 = st.columns([2,1])
            with col1:
                st.markdown("###### Deployed Models")
            with col2:
                st.form_submit_button("Manage")
            
            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row")
    else:
        st.write("There are no deployed models")


#load data for page
with st.spinner("Loading data"):
    if "available_protein_models_df" not in st.session_state:
            available_protein_models_df = get_available_models(ModelCategory.PROTEIN_STUDIES, get_app_context())
            available_protein_models_df["model_labels"] = (available_protein_models_df["model_id"].astype(str) + " - " 
                                                + available_protein_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_protein_models_df["model_uc_name"].astype(str) + " v"
                                                + available_protein_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_protein_models_df"] = available_protein_models_df
    available_protein_models_df = st.session_state["available_protein_models_df"]

    if "deployed_protein_models_df" not in st.session_state:
        deployed_protein_models_df = get_deployed_models(ModelCategory.PROTEIN_STUDIES, get_app_context())
        deployed_protein_models_df.columns = ["Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version"]

        st.session_state["deployed_protein_models_df"] = deployed_protein_models_df
    deployed_protein_models_df = st.session_state["deployed_protein_models_df"]

settings, esm, alpha = st.tabs(["Settings","Protein Design", "Protein Folding"])

with settings:
    display_protein_studies_settings(available_protein_models_df, deployed_protein_models_df)