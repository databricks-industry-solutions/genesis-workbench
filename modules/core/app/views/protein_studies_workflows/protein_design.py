"""Protein Studies — Protein Design tab: ESMFold + RFDiffusion + ProteinMPNN pipeline."""

import streamlit as st
import streamlit.components.v1 as components
import traceback
import logging
from datetime import datetime

from utils.molstar_tools import molstar_html_multibody
from utils.protein_design import make_designs, align_designed_pdbs
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from genesis_workbench.models import MLflowExperimentAccessException


def _get_progress_callback(status_generation):
    def report_progress(progress_report: dict):
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


def _design_tab_fn(sequence: str, mlflow_experiment: str, mlflow_run_name: str, progress_callback=None) -> str:
    user_info = get_user_info()
    n_rf_diffusion = 1
    logging.info("design: make designs")
    output = make_designs(
        sequence=sequence,
        mlflow_experiment_name=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        user_info=user_info,
        n_rfdiffusion_hits=n_rf_diffusion,
        progress_callback=progress_callback)

    designed_pdbs = {"initial": output["initial"], "designed": output["designed"]}
    logging.info("design: align")
    aligned_structures = align_designed_pdbs(designed_pdbs)
    logging.info("design: get html for designs")
    html = molstar_html_multibody(aligned_structures)
    return html, output['experiment_id'], output['run_id']


def render():
    st.markdown("###### Protein Structure Design with ESMfold, RFDiffusion and ProteinMPNN")
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with c1:
        gen_input_sequence = st.text_area(
            "Provide an input sequence where the region between square braces is to be replaced/in-painted by new designs:",
            "MAQVKLQESGGGLVQPGGSLRLSCASSVPIFAITVMGWYRQAPGKQRELVAGIKRSGD[TNYADS]VKGRFTISRDDAKNTVFLQMNSLTTEDTAVYYCNAQILSWMGGTDYWGQGTQVTVSSGQAGQ",
            height=180, help="Example: `CASRRSG[FTYPGF]FFEQYF`")

    with c2:
        protein_design_mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_protein_design")
        _ts = datetime.now().strftime("%Y%m%d_%H%M")
        protein_design_mlflow_run = st.text_input("Run Name:", value=f"protein_design_{_ts}")

        c11, c12, c13 = st.columns([1, 1, 1])
        with c11:
            generate_btn = st.button("Generate")
        with c12:
            clear_btn = st.button("Clear", key="clear_gen_btn")

    mol_viewer = st.container()
    if generate_btn:
        is_valid = True
        if gen_input_sequence.strip() == "" or "[" not in gen_input_sequence or "]" not in gen_input_sequence:
            is_valid = False
            st.error("Enter a valid sequence with the region to be replaced marked by square braces")

        if protein_design_mlflow_experiment.strip() == "" or protein_design_mlflow_run.strip() == "":
            is_valid = False
            st.error("Enter an mlflow experiment and run name")

        if is_valid:
            with st.spinner("Generating.."):
                with mol_viewer:
                    status_generation = st.progress(0, text="Generating Sequence")
                    try:
                        html, experiment_id, run_id = _design_tab_fn(
                            sequence=gen_input_sequence,
                            mlflow_experiment=protein_design_mlflow_experiment,
                            mlflow_run_name=protein_design_mlflow_run,
                            progress_callback=_get_progress_callback(status_generation))

                        st.button("View MLflow Experiment", on_click=lambda: open_mlflow_experiment_window(experiment_id))
                        components.html(html, height=700)
                    except MLflowExperimentAccessException:
                        st.error("Cannot access MLflow folder. Please complete MLflow Setup in the profile page")
                    except Exception as e:
                        st.error(f"An error occured while generating the sequence: {e}")
                        traceback.print_exc()

    if clear_btn:
        mol_viewer.empty()
