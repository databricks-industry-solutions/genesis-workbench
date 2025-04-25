
import streamlit as st
import pandas as pd
import time

from genesis_workbench.models import ModelCategory, get_available_models, get_deployed_models
from genesis_workbench.models import ModelCategory, get_uc_model_info, import_model_from_uc

@st.dialog("Import model from Unity Catalog")
def display_import_model_uc_dialog():    
    
    model_info = None
    model_info_error = False
    model_import_error = False
    fetch_model_info_clicked = False
    uc_import_model_clicked = False

    if "import_uc_model_info" in st.session_state:
        model_info = st.session_state["import_uc_model_info"]

    with st.form("import_model_uc_form_fetch", enter_to_submit=False ):
        c1,c2,c3 = st.columns([3,1,1], vertical_alignment="bottom")
        with c1:
            uc_model_name = st.text_input("Unity Catalog Name (catalog.schema.model_name):", value="genesis_workbench.dev_srijit_nair_dbx_genesis_workbench_core.test_model")
        with c2:
            uc_model_version = st.number_input("Version:", min_value=1, step=1, max_value=999)
        with c3:
            fetch_model_info_clicked = st.form_submit_button(":material/refresh:")
    
    if fetch_model_info_clicked:
        with st.spinner("Getting model info"):
            try:
                model_info = None
                model_info = get_uc_model_info(uc_model_name, uc_model_version)
                st.session_state["import_uc_model_info"] = model_info
            except Exception as e:                    
                model_info_error = True
                del st.session_state["import_uc_model_info"]
    
    if(model_info_error):
        st.error("Error fetching model details.")        
            
    if model_info:
        with st.form("import_model_uc_form_import", enter_to_submit=False):
            model_name = st.text_input("Model Name:", value=uc_model_name.split(".")[2], help="Common name of the mode if different from UC name.")
            model_source_version = st.text_input("Source Model Version:" , help="Source version of the corresponding UC model.")
            model_display_name = st.text_input("Display Name:",value=uc_model_name.split(".")[2], help="Name that will be displayed on UI.")
            model_description_url = st.text_input("Description URL:", help="A website URL where users can read more about this model")
            
            if st.form_submit_button('Import Model'):
                uc_import_model_clicked = True

    if uc_import_model_clicked:
        with st.spinner("Importing model"):
            try:
                import_model_from_uc(model_category = ModelCategory.SINGLE_CELL,
                    model_uc_name = uc_model_name,
                    model_uc_version =  uc_model_version, 
                    model_name = model_name,
                    model_source_version = model_source_version,
                    model_display_name = model_display_name,
                    model_description_url = model_description_url)
                
                model_info = None
                del st.session_state["import_uc_model_info"]
            except Exception as e:
                model_import_error = True

    if uc_import_model_clicked:
        if model_import_error:
            st.error("Error importing model") 
        else:
            st.success("Model Imported Successfully. Refreshing data..")
            time.sleep(3)
            del st.session_state["available_models_df"]
            st.rerun()
            #if st.button("Close"):
            #    if "import_button" in st.session_state:
            #        del st.session_state["import_button"]              
            #    st.rerun()


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
                display_import_model_uc_dialog()


        st.markdown("###### Available Models:")
        with st.form("deploy_model_form"):
            col1, col2, = st.columns([1,1])    
            with col1:
                select_models = st.selectbox("Model:",available_models_df["model_labels"],label_visibility="collapsed",)

            with col2:
                deploy_button = st.form_submit_button('Deploy')


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
    if "available_models_df" not in st.session_state:
            available_models_df = get_available_models(ModelCategory.SINGLE_CELL)
            available_models_df["model_labels"] = (available_models_df["model_id"].astype(str) + " - " 
                                                + available_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_models_df["model_uc_name"].astype(str) + " v"
                                                + available_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_models_df"] = available_models_df
    available_models_df = st.session_state["available_models_df"]

    if "deployed_models_df" not in st.session_state:
        deployed_models_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        deployed_models_df.columns = ["Id", "Name", "Source Version", "UC Name", "UC Version", "Deployment Ids"]

        st.session_state["deployed_models_df"] = deployed_models_df
    deployed_models_df = st.session_state["deployed_models_df"]



st.title(":material/microbiology:  Single Cell Studies")

settings_tab, embeddings_tab = st.tabs(["Settings","Embeddings"])

with settings_tab:
    display_settings_tab(available_models_df,deployed_models_df)

with embeddings_tab:
    display_embeddings_tab(deployed_models_df)
