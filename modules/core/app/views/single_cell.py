
import streamlit as st
import pandas as pd
import time
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models)

from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog)

def reset_available_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["available_single_cell_models_df"]
        st.rerun()

def reset_deployed_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["deployed_single_cell_models_df"]
        st.rerun()

def display_settings_tab(available_models_df,deployed_models_df):

    p1,p2 = st.columns([2,1])

    with p1:
        st.markdown("###### Import Models:")
        with st.form("import_model_form"):
            col1, col2, = st.columns([1,1], vertical_alignment="bottom")    
            with col1:
                import_model_source = st.selectbox("Source:",["Unity Catalog","Hugging Face","PyPi"],label_visibility="visible")

            with col2:
                import_button = st.form_submit_button('Import')
        
        if import_button:
            if import_model_source=="Unity Catalog":
                display_import_model_uc_dialog(ModelCategory.SINGLE_CELL, success_callback=reset_available_models)


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


def display_embeddings_tab(deployed_models_df):
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("###### Generate Embeddings")
    with col2:        
        st.button("View Past Runs")    

    if len(deployed_models_df) > 0:
        with st.form("run_embedding_form"):
            st.write("Select Models:")


            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="multi-row")
        
            st.write("NOTE: A result table will be created for EACH model selected.")

            col1, col2, col3 = st.columns([1,1,1], vertical_alignment="bottom")
            with col1:        
                st.text_input("Data Location:","")
                st.text_input("Result Schema Name:","")
                st.text_input("Result Table Prefix:","")
            
            with col2:
                st.write("")
                st.toggle("Perform Evaluation?")            
                st.text_input("Ground Truth Data Location:","")
                st.text_input("MLflow Experiment Name:","")
            
            st.form_submit_button("Generate Embeddings")

    else:
        st.write("There are no deployed models")

#load data for page
with st.spinner("Loading data"):
    if "available_single_cell_models_df" not in st.session_state:
            available_single_cell_models_df = get_available_models(ModelCategory.SINGLE_CELL)
            available_single_cell_models_df["model_labels"] = (available_single_cell_models_df["model_id"].astype(str) + " - " 
                                                + available_single_cell_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_single_cell_models_df["model_uc_name"].astype(str) + " v"
                                                + available_single_cell_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_single_cell_models_df"] = available_single_cell_models_df
    available_single_cell_models_df = st.session_state["available_single_cell_models_df"]

    if "deployed_single_cell_models_df" not in st.session_state:
        deployed_single_cell_models_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        deployed_single_cell_models_df.columns = ["Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version"]

        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]



st.title(":material/microbiology:  Single Cell Studies")

settings_tab, embeddings_tab = st.tabs(["Settings","Embeddings"])

with settings_tab:
    display_settings_tab(available_single_cell_models_df,deployed_single_cell_models_df)

with embeddings_tab:
    display_embeddings_tab(deployed_single_cell_models_df)
