import streamlit as st
import traceback
import os
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models, 
                                      MLflowExperimentAccessException)

from utils.streamlit_helper import (get_user_info, 
                                    open_run_window,
                                    display_deploy_model_dialog,
                                    open_mlflow_experiment_window)

from utils.molstar_tools import (
                    html_as_iframe,
                    molstar_html_singlebody,
                    molstar_html_multibody)

from utils.protein_structure import start_run_alphafold_job

from utils.protein_design import (make_designs, 
                                  hit_esmfold,
                                  align_designed_pdbs)

import streamlit.components.v1 as components

import logging

st.title(":material/biotech: Protein Studies")

def get_progress_callback(status_generation
        #status_parsing,status_esm_init,status_rfdiffusion,status_proteinmpnn,status_esm_preds
        ) :
    def report_progress (progress_report: dict):
        # status_parsing.progress(progress_report["status_parsing"], text="Parsing Sequence")
        # status_esm_init.progress(progress_report["status_esm_init"], text="Generating structure using ESMFold")
        # status_rfdiffusion.progress(progress_report["status_rfdiffusion"], text="Generating protein using RFdiffusion")
        # status_proteinmpnn.progress(progress_report["status_proteinmpnn"], text="Predicting sequences using ProteinMPNN")
        # status_esm_preds.progress(progress_report["status_esm_preds"],text="Generating structure for new protein using ESMFold")
        status_text = ""
        if progress_report["status_parsing"] < 100:
            status_text = "Parsing Sequence"
            progress = 20
        elif progress_report["status_esm_init"] < 100:
            status_text = "Generating original structure using ESMFold"
            progress = 40
        elif progress_report["status_rfdiffusion"] < 100:
            status_text = "Generating protein backbone for the new region using RFdiffusion"
            progress = 60
        elif progress_report["status_proteinmpnn"] < 100:
            status_text = "Infering sequences of backbones using ProteinMPNN"
            progress = 80
        elif progress_report["status_esm_preds"] < 100:
            status_text = "Generating new protein using ESMFold and aligning to original"
            progress = 90
        else:
            status_text = "Generation complete"
            progress = 100
        status_generation.progress(progress, status_text)

    return report_progress

def esmfold_btn_fn(protein : str) -> str:
    pdb = hit_esmfold(protein)
    html =  molstar_html_multibody(pdb)
    return html

def design_tab_fn(sequence: str, mlflow_experiment:str, mlflow_run_name:str, progress_callback=None) -> str:
    user_info = get_user_info()    
    n_rf_diffusion: int = 1
    logging.info("design: make designs")
    output = make_designs(sequence=sequence, 
                                 mlflow_experiment_name=mlflow_experiment,
                                 mlflow_run_name=mlflow_run_name,
                                 user_info=user_info,
                                 n_rfdiffusion_hits=n_rf_diffusion,
                                 progress_callback=progress_callback)
    
    designed_pdbs = {"initial" : output["initial"],
                     "designed" : output["designed"] }
    
    logging.info("design: align")    
    # logging.info([k for k in designed_pdbs.keys()])
    # logging.info([v[:10] for v in designed_pdbs.values()])
    aligned_structures = align_designed_pdbs(designed_pdbs)
    logging.info("design: get html for designs")           
    html =  molstar_html_multibody(aligned_structures)
    return html , output['experiment_id'],  output['run_id']


def display_protein_studies_settings(available_models_df,deployed_models_df):

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
            available_protein_models_df = get_available_models(ModelCategory.PROTEIN_STUDIES)
            available_protein_models_df["model_labels"] = (available_protein_models_df["model_id"].astype(str) + " - " 
                                                + available_protein_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_protein_models_df["model_uc_name"].astype(str) + " v"
                                                + available_protein_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_protein_models_df"] = available_protein_models_df
    available_protein_models_df = st.session_state["available_protein_models_df"]

    if "deployed_protein_models_df" not in st.session_state:
        deployed_protein_models_df = get_deployed_models(ModelCategory.PROTEIN_STUDIES)
        deployed_protein_models_df.columns = ["Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version"]

        st.session_state["deployed_protein_models_df"] = deployed_protein_models_df
    deployed_protein_models_df = st.session_state["deployed_protein_models_df"]

settings_tab, protein_structure_prediction_tab, protein_design_tab = st.tabs(["Settings", "Protein Structure Prediction", "Protein Design"])

with settings_tab:
    display_protein_studies_settings(available_protein_models_df, deployed_protein_models_df)


with protein_structure_prediction_tab:
    st.markdown("###### Predict Protein Structure")

    view_model_choice =  st.pills("Model:",["ESMFold","AlphaFold2"], 
                                selection_mode="single",
                                default="ESMFold") 

    if view_model_choice=="ESMFold":
        c1,c2,c3 = st.columns([3,1,1], vertical_alignment="bottom")
        with c1:
            view_esmfold_input_sequence = st.text_area("Provide an input sequence to infer the structure:"
                                        ,"MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE", key="view_esmfold_input_sequence")
        with c2:
            view_structure_esmfold_btn = st.button("View", key="view_structure_esmfold_btn")
            clear_view_esmfold_btn = st.button("Clear", key="clear_view_esmfold_btn")

        prot_viewer = st.container()
        if view_structure_esmfold_btn:
            with st.spinner("Generating structure.."):
                with prot_viewer:
                    html =  esmfold_btn_fn(view_esmfold_input_sequence)
                    components.html(html, height=700)
                            
        if clear_view_esmfold_btn:
            prot_viewer.empty()
             

    if view_model_choice=="AlphaFold2":
        st.write("Runs a workflow to do MSA and template search and then perform structure prediction")
        c1,c2 = st.columns([3,1], vertical_alignment="bottom")
        with c1:
            view_alphafold_input_sequence = st.text_area("Provide an input sequence to infer the structure:"
                                        ,"QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS", key="view_alphafold_input_sequence")
        
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="bottom")
        with c1:
            view_alphafold_run_experiment = st.text_input("MLflow Experiment:","",placeholder="structure_prediction_alphafold")            
        with c2:
            view_alphafold_run_label = st.text_input("Run Name:","",placeholder="my_run_123")
        with c3:
            view_structure_alphafold_btn = st.button("Start Job", key="view_structure_alphafold_btn")

        if view_structure_alphafold_btn:
            is_valid = True
            if view_alphafold_input_sequence.strip() == "" :
                is_valid = False
                st.error("Enter a valid sequence with the region to be replaced marked by square braces")

            if (view_alphafold_run_experiment.strip() == ""  or 
                view_alphafold_run_label.strip() == ""):
                is_valid = False
                st.error("Enter an mlflow experiment and run name")

            if is_valid:    
                user_info = get_user_info()  
                try:
                    with st.spinner("Starting job"):
                        alphafold_job_run_id = start_run_alphafold_job(protein_sequence=view_alphafold_input_sequence,
                                    mlflow_experiment_name=view_alphafold_run_experiment,
                                    mlflow_run_name=view_alphafold_run_label,
                                    user_info=user_info)
                        
                        st.success(f"Job started with run id: {alphafold_job_run_id}.")                
                        run_alphafold_job_id = os.getenv("RUN_ALPHAFOLD_JOB_ID")
                        view_deploy_run_btn = st.button("View Run", on_click=lambda: open_run_window(run_alphafold_job_id,alphafold_job_run_id))

                except Exception as e:
                    st.error("An error occured while running the workflow")
                    print(e)
                


        st.divider()
        st.markdown("###### Search Past Runs:")
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="bottom")

        with c1:
            search_alphafold_run_experiment = st.text_input("MLflow Experiment:","",key="search_alphafold_run_experiment")            
        with c2:
            search_alphafold_run_name = st.text_input("Run Name:","", key="search_alphafold_run_name")
        with c3:
            search_alphafold_run_button= st.button("Search", key="search_alphafold_run_button")

    


with protein_design_tab:
    st.markdown("###### Protein Structure Design with ESMfold, RFDiffusion and ProteinMPNN")
    c1,c2,c3 = st.columns([2,1,1], vertical_alignment="bottom")
    with c1:
        gen_input_sequence = st.text_area("Provide an input sequence where the region between square braces is to be replaced/in-painted by new designs:"
                                      ,"MAQVKLQESGGGLVQPGGSLRLSCASSVPIFAITVMGWYRQAPGKQRELVAGIKRSGD[TNYADS]VKGRFTISRDDAKNTVFLQMNSLTTEDTAVYYCNAQILSWMGGTDYWGQGTQVTVSSGQAGQ"
                                      , height=180, help="Example: `CASRRSG[FTYPGF]FFEQYF`")
    
    with c2:
        protein_design_mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_protein_design")
        protein_design_mlflow_run = st.text_input("Run Name:")

        c11,c12, c13 = st.columns([1,1,1])
        with c11:
            generate_btn = st.button("Generate")

        with c12:
            clear_btn = st.button("Clear", key="clear_gen_btn")

    mol_viewer = st.container()
    if generate_btn:
        is_valid = True
        if (gen_input_sequence.strip() == "" or 
            "[" not in gen_input_sequence or
            "]" not in gen_input_sequence):
            is_valid = False
            st.error("Enter a valid sequence with the region to be replaced marked by square braces")

        if (protein_design_mlflow_experiment.strip() == ""  or 
            protein_design_mlflow_run.strip() == ""):
            is_valid = False
            st.error("Enter an mlflow experiment and run name")

        if is_valid:

            with st.spinner("Generating.."):
                with mol_viewer:
                    # status_parsing = st.progress(0, text="Parsing Sequence")
                    # status_esm_init = st.progress(0, text="Generating structure using ESMFold")
                    # status_rfdiffusion = st.progress(0, text="Generating protein using RFdiffusion")
                    # status_proteinmpnn = st.progress(0, text="Predicting sequences using ProteinMPNN")
                    # status_esm_preds = st.progress(0, text="Generating structure for new protein using ESMFold")
                    status_generation = st.progress(0, text="Generating Sequence")

                    try:
                        html, experiment_id, run_id = design_tab_fn(sequence=gen_input_sequence, 
                                            mlflow_experiment=protein_design_mlflow_experiment,
                                            mlflow_run_name=protein_design_mlflow_run,
                                            progress_callback=get_progress_callback(status_generation))
                    
                        view_mlflow_experiment_btn = st.button("View MLflow Experiment", on_click=lambda: open_mlflow_experiment_window(experiment_id))
                        components.html(html, height=700)
                    except MLflowExperimentAccessException as mle:
                        st.error("Cannot access MLflow folder. Please complete MLflow Setup in the profile page")
                    except Exception as e:
                        st.error(f"An error occured while generating the sequence: {e}")
                        traceback.print_exc()


            
    if clear_btn:
        mol_viewer.empty()
    
        

