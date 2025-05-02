
import streamlit as st
import pandas as pd
import time
import os
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models,
                                      get_uc_model_info,
                                      import_model_from_uc,
                                      get_gwb_model_info,
                                      deploy_model)
from genesis_workbench.workbench import UserInfo

from streamlit.components.v1 import html


def get_user_info():
    headers = st.context.headers
    user_access_token = headers.get("X-Forwarded-Access-Token")
    user_name=headers.get("X-Forwarded-Preferred-Username")
    user_display_name = ""
    if user_access_token:
        # Initialize WorkspaceClient with the user's token
        w = WorkspaceClient(token=user_access_token, auth_type="pat")
        # Get current user information
        current_user = w.current_user.me()
        # Display user information
        user_display_name = current_user.display_name

    return UserInfo(
        user_name=user_name,
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        user_access_token = headers.get("X-Forwarded-Access-Token"),
        user_display_name = user_display_name if user_display_name != "" else user_name
    )

def open_deploy_model_run_window(run_id):
    host_name = os.getenv("DATABRICKS_HOST")
    job_id = os.getenv("DEPLOY_MODEL_JOB_ID")
    url = f"https://{host_name}#job/{job_id}/run/{run_id}"
    print(url)
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

@st.dialog("Deploy Model", width="large")
def display_deploy_model_dialog(selected_model_name):    
    """Dialog to deploy a model to model serving"""
    model_info = None
    run_id = None
    deploy_model_clicked = False
    view_deploy_run_btn = False
    close_deploy_run_btn = False
    user_info = get_user_info()
    
    model_id = int(selected_model_name.split("-")[0].strip())

    
    if "deployment_model_details" in st.session_state:
        model_info = st.session_state["deployment_model_details"]
    else:
        with st.spinner("Getting model details"):
            try:
                model_info = get_gwb_model_info(model_id)        
                st.session_state["deployment_model_details"] = model_info
            except Exception as e:
                st.error("Error getting model details.")
                model_info = None

    if model_info:
        model_details = model_info.model_uc_name.split(".") 
        st.write(f"Model Name: {model_info.model_display_name} ")
        st.write(f"Registered Name: {model_info.model_uc_name} v{model_info.model_uc_version}")
        
        if model_info.is_model_deployed:
            st.warning("❗️This model has existing deployment(s).")

        with st.form("deploy_model_details_form", enter_to_submit=False):
            deploy_name = st.text_input("Deployment Name:", placeholder="eg: finetuned geneformer")
            deploy_description= st.text_area("Deployment Description:", max_chars=5000)
            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                st.write("Input Schema")
                st.json(model_info.model_input_schema)
            with c2:
                st.write("Output Schema")
                st.json(model_info.model_output_schema)
            with c3:
                st.write("Parameters")
                st.json(model_info.model_params_schema)
            
            c1,c2 = st.columns([1,1])
            with c1:
                input_adapter_code = st.file_uploader("Input Adapter:", type="py", help="A python file with only one class definition that extends `genesis_workbench.models.BaseAdapter`." )
            with c2: 
                output_adapter_code = st.file_uploader("Output Adapter:", type="py", help="A python file with only one class definition that extends `genesis_workbench.models.BaseAdapter`." )
            
            new_sample_input = st.text_area("Provide a sample input data (required if using adapters):")

            compute_type = st.selectbox("Model Serving Compute:", ["CPU", "GPU SMALL", "GPU MEDIUM", "GPU LARGE"])
            workload_size = st.selectbox("Workload Size:", ["Small", "Medium","Large"])            
            deploy_model_clicked = st.form_submit_button("Deploy Model")

        deploy_started = False
        if deploy_model_clicked:
            with st.spinner("Launching deploy job"):
                try:
                    run_id = deploy_model(user_info = user_info,
                                          gwb_model_id = model_id,
                                          deployment_name=deploy_name,
                                          deployment_description=deploy_description, 
                                          workload_type=compute_type,
                                          workload_size=workload_size)

                    deploy_started = True
                except Exception as e:
                    print(e)
                    st.error("Error launching deploy job.")                
                    deploy_started = False
        if deploy_started:
            st.success(f"Model deploy has started with a run id {run_id}.")                
            st.warning(f"It might take upto 30 minutes to complete")
            view_deploy_run_btn = st.button("View Run", on_click=lambda: open_deploy_model_run_window(run_id))


@st.dialog("Import model from Unity Catalog")
def display_import_model_uc_dialog():    
    """Dialog to import a model from UC"""
    model_info = None
    model_info_error = False
    model_import_error = False
    fetch_model_info_clicked = False
    uc_import_model_clicked = False
    user_info = get_user_info()

    if "import_uc_model_info" in st.session_state:
        model_info = st.session_state["import_uc_model_info"]

    with st.form("import_model_uc_form_fetch", enter_to_submit=False ):
        c1,c2,c3 = st.columns([3,1,1], vertical_alignment="bottom")
        with c1:
            uc_model_name = st.text_input("Unity Catalog Name (catalog.schema.model_name):", value="genesis_workbench.dev_srijit_nair_dbx_genesis_workbench_core.gene_embedder")
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
                import_model_from_uc(user_info = user_info,
                    model_category = ModelCategory.SINGLE_CELL,
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
            st.success("Model Imported Successfully.")
            with st.spinner("Refreshing data.."):
                time.sleep(1)
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
        deployed_models_df.columns = ["Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version"]

        st.session_state["deployed_models_df"] = deployed_models_df
    deployed_models_df = st.session_state["deployed_models_df"]



st.title(":material/microbiology:  Single Cell Studies")

settings_tab, embeddings_tab = st.tabs(["Settings","Embeddings"])

with settings_tab:
    display_settings_tab(available_models_df,deployed_models_df)

with embeddings_tab:
    display_embeddings_tab(deployed_models_df)
