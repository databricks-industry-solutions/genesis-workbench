
import streamlit as st
import pandas as pd
import time
import os
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models)

from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog,
                                    get_user_info,
                                    open_run_window)
from utils.single_cell_analysis import start_scanpy_job, start_rapids_singlecell_job

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
                            selection_mode="single-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
    else:
        st.write("There are no deployed models")


def display_scanpy_analysis_tab():
    st.markdown("###### Run Scanpy Analysis")
    
    with st.form("scanpy_analysis_form"):
        # Mode Selection
        st.markdown("**Analysis Mode:**")
        mode = st.selectbox(
            "Mode",
            options=["scanpy", "rapids-singlecell"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Data Input
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Data Configuration:**")
            data_path = st.text_input(
                "Data Path (h5ad file)",
                placeholder="/Volumes/catalog/schema/volume/file.h5ad",
                help="Path to the h5ad file in Unity Catalog Volumes"
            )
            gene_name_column = st.text_input(
                "Gene Name Column",
                value="gene_name",
                help="Column name containing gene names"
            )
            
            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input(
                "MLflow Experiment Name",
                value="scanpy_genesis_workbench",
                help="Simple experiment name (will be created in your MLflow folder)"
            )
            mlflow_run_name = st.text_input(
                "MLflow Run Name",
                value="scanpy_analysis",
                help="Name for this specific analysis run"
            )
        
        with col2:
            st.markdown("**Filtering Parameters:**")
            min_genes = st.number_input(
                "Min Genes per Cell",
                min_value=0,
                value=200,
                step=10
            )
            min_cells = st.number_input(
                "Min Cells per Gene",
                min_value=0,
                value=3,
                step=1
            )
            pct_counts_mt = st.number_input(
                "Max % Mitochondrial Counts",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1
            )
            n_genes_by_counts = st.number_input(
                "Max Genes by Counts",
                min_value=0,
                value=2500,
                step=100
            )
        
        st.divider()
        
        # Analysis Parameters
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("**Normalization & Feature Selection:**")
            target_sum = st.number_input(
                "Target Sum for Normalization",
                min_value=0,
                value=10000,
                step=1000
            )
            n_top_genes = st.number_input(
                "Number of Highly Variable Genes",
                min_value=0,
                value=500,
                step=50
            )
        
        with col4:
            st.markdown("**Dimensionality Reduction & Clustering:**")
            n_pcs = st.number_input(
                "Number of Principal Components",
                min_value=0,
                value=50,
                step=5
            )
            leiden_resolution = st.number_input(
                "Leiden Resolution",
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.1,
                format="%.1f"
            )
        
        st.divider()
        submit_button = st.form_submit_button("Start Analysis", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        is_valid = True
        
        # Validation
        if not data_path.strip():
            st.error("Please provide a data path")
            is_valid = False
        elif not data_path.endswith(".h5ad"):
            st.error("Data path must point to an .h5ad file")
            is_valid = False
        elif not data_path.startswith("/Volumes"):
            st.error("Data path must start with /Volumes (Unity Catalog Volume)")
            is_valid = False
        
        if not mlflow_experiment.strip() or not mlflow_run_name.strip():
            st.error("Please provide MLflow experiment and run names")
            is_valid = False
        
        if is_valid:
            user_info = get_user_info()
            
            if mode == "scanpy":
                try:
                    with st.spinner("Starting scanpy analysis job..."):
                        scanpy_job_id, job_run_id = start_scanpy_job(
                            data_path=data_path,
                            mlflow_experiment=mlflow_experiment,
                            mlflow_run_name=mlflow_run_name,
                            gene_name_column=gene_name_column,
                            min_genes=min_genes,
                            min_cells=min_cells,
                            pct_counts_mt=pct_counts_mt,
                            n_genes_by_counts=n_genes_by_counts,
                            target_sum=target_sum,
                            n_top_genes=n_top_genes,
                            n_pcs=n_pcs,
                            leiden_resolution=leiden_resolution,
                            user_info=user_info
                        )
                        
                        # Construct the run URL
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/{scanpy_job_id}/runs/{job_run_id}"
                        
                        st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                        st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                
                except Exception as e:
                    st.error(f"❌ An error occurred while starting the job: {str(e)}")
                    print(e)
            
            elif mode == "rapids-singlecell":
                st.warning("⚠️ rapids-singlecell mode is not yet implemented. Coming soon!")
                # TODO: When rapids-singlecell job is deployed, replace the warning above with:
                # try:
                #     with st.spinner("Starting rapids-singlecell analysis job..."):
                #         rapids_job_id, job_run_id = start_rapids_singlecell_job(
                #             data_path=data_path,
                #             mlflow_experiment=mlflow_experiment,
                #             mlflow_run_name=mlflow_run_name,
                #             gene_name_column=gene_name_column,
                #             min_genes=min_genes,
                #             min_cells=min_cells,
                #             pct_counts_mt=pct_counts_mt,
                #             n_genes_by_counts=n_genes_by_counts,
                #             target_sum=target_sum,
                #             n_top_genes=n_top_genes,
                #             n_pcs=n_pcs,
                #             leiden_resolution=leiden_resolution,
                #             user_info=user_info
                #         )
                #         
                #         st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                #         st.button("View Run", on_click=lambda: open_run_window(rapids_job_id, job_run_id))
                # 
                # except Exception as e:
                #     st.error(f"❌ An error occurred while starting the job: {str(e)}")
                #     print(e)
            
            else:
                st.error(f"❌ Unknown mode: {mode}")


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
                            selection_mode="multi-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
        
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
        deployed_single_cell_models_df.columns = ["Model Id","Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]

        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]



st.title(":material/microbiology:  Single Cell Studies")

settings_tab, scanpy_tab, embeddings_tab = st.tabs(["Settings", "Scanpy Analysis", "Embeddings"])

with settings_tab:
    display_settings_tab(available_single_cell_models_df,deployed_single_cell_models_df)

with scanpy_tab:
    display_scanpy_analysis_tab()

with embeddings_tab:
    display_embeddings_tab(deployed_single_cell_models_df)
