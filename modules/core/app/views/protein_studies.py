import streamlit as st
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models,                                      
                                      get_gwb_model_info,
                                      deploy_model)

from utils.streamlit_helper import (get_app_context, 
                                    display_import_model_uc_dialog,
                                    display_deploy_model_dialog)

from utils.molstar_tools import (
                    html_as_iframe,
                    molstar_html_singlebody,
                    molstar_html_multibody)

from utils.protein_design import (make_designs, 
                                  hit_esmfold,
                                  align_designed_pdbs)

import streamlit.components.v1 as components

import logging

st.title(":material/biotech: Protein Structure Prediction")

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

def design_tab_fn(sequence: str, progress_callback=None) -> str:
    
    n_rf_diffusion: int = 1
    logging.info("design: make designs")
    designed_pdbs = make_designs(sequence, progress_callback=progress_callback)
    logging.info("design: align")
    print(designed_pdbs)
    # logging.info([k for k in designed_pdbs.keys()])
    # logging.info([v[:10] for v in designed_pdbs.values()])
    aligned_structures = align_designed_pdbs(designed_pdbs)
    logging.info("design: get html for designs")           
    html =  molstar_html_multibody(aligned_structures)
    return html


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
        
        c1,c2,c3,c4 = st.columns([1,1,1,3], vertical_alignment="bottom")
        with c1:
            view_alphafold_run_label = st.text_input("Run label:","",placeholder="my_run_123")
        with c2:
            view_alphafold_run_experiment = st.text_input("MLflow Experiment:","",placeholder="structure_prediction_alphafold")
        with c3:
            view_structure_alphafold_btn = st.button("Start Job", key="view_structure_alphafold_btn")
        
        st.divider()
        st.markdown("###### Search Past Runs:")
        c1,c2,c3 = st.columns([1,1,3], vertical_alignment="bottom")
        with c1:
            search_alphafold_run_label = st.text_input("Run label:","",placeholder="my_run_123", key="search_alphafold_run_label")
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
        protein_design_mlflow_experiment = st.text_input("MLflow Experiment:")
        protein_design_mlflow_run = st.text_input("Run Name:")

        c11,c12, c13 = st.columns([1,1,1])
        with c11:
            generate_btn = st.button("Generate")

        with c12:
            clear_btn = st.button("Clear", key="clear_gen_btn")

    mol_viewer = st.container()
    if generate_btn:
        with st.spinner("Generating.."):
            with mol_viewer:
                # status_parsing = st.progress(0, text="Parsing Sequence")
                # status_esm_init = st.progress(0, text="Generating structure using ESMFold")
                # status_rfdiffusion = st.progress(0, text="Generating protein using RFdiffusion")
                # status_proteinmpnn = st.progress(0, text="Predicting sequences using ProteinMPNN")
                # status_esm_preds = st.progress(0, text="Generating structure for new protein using ESMFold")
                status_generation = st.progress(0, text="Generating Sequence")

                html =  design_tab_fn(gen_input_sequence, progress_callback=get_progress_callback(
                    # status_parsing,
                    # status_esm_init,
                    # status_rfdiffusion,
                    # status_proteinmpnn,
                    # status_esm_preds
                    status_generation
                ))
                components.html(html, height=700)
            
    if clear_btn:
        mol_viewer.empty()
    
        

