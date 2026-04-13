"""Protein Studies — Structure Prediction tab: ESMFold and AlphaFold2."""

import streamlit as st
import streamlit.components.v1 as components
import os
import logging
from datetime import datetime

from utils.molstar_tools import molstar_html_multibody
from utils.protein_design import hit_esmfold, hit_boltz
from utils.protein_structure import (start_run_alphafold_job,
                                     search_alphafold_runs_by_run_name,
                                     search_alphafold_runs_by_experiment_name,
                                     af_collect_and_align)
from utils.streamlit_helper import get_user_info, open_run_window
from typing import Optional


def _esmfold_btn_fn(protein: str) -> str:
    pdb = hit_esmfold(protein)
    return molstar_html_multibody(pdb)


def _view_structure_from_alphafold_run(run_id: str, run_name: str, pdb_code: Optional[str] = None, include_pdb: bool = False) -> str:
    logging.info('running alphafold viewer')
    pdb_run, true_structure_str, af_structure_str = af_collect_and_align(
        run_id=run_id, run_name=run_name, pdb_code=pdb_code, include_pdb=include_pdb
    )
    if include_pdb:
        return molstar_html_multibody([af_structure_str, true_structure_str])
    return molstar_html_multibody(af_structure_str)


def _set_selected_row_status():
    selection = st.session_state["alphafold_run_search_result_display_df"].selection
    if len(selection["rows"]) > 0:
        selected_index = selection["rows"][0]
        selected_alphafold_run_status = st.session_state["alphafold_run_search_result_df"].iloc[selected_index]["status"]
        st.session_state["selected_alphafold_run_status"] = selected_alphafold_run_status
    else:
        if "selected_alphafold_run_status" in st.session_state:
            del st.session_state["selected_alphafold_run_status"]


@st.dialog("View Structure", width="large")
def _display_view_alphafold_result_dialog(selected_row_for_view):
    run_id = st.session_state["alphafold_run_search_result_df"].iloc[selected_row_for_view]["run_id"].iloc[0]
    run_name = st.session_state["alphafold_run_search_result_df"].iloc[selected_row_for_view]["run_name"].iloc[0]
    st.markdown(f"##### Run Name: {run_name}")
    include_pdb = False
    pdb_to_compare = None

    pdb_compare_c1, pdb_compare_c2 = st.columns([2, 1], vertical_alignment="bottom")
    with pdb_compare_c1:
        af_run_compare_pbm_id = st.text_input("PDB Code: ")
    with pdb_compare_c2:
        compare_pdb_btn = st.button("Compare")

    if len(af_run_compare_pbm_id.strip()) > 0:
        include_pdb = True
        pdb_to_compare = af_run_compare_pbm_id.strip()

    with st.spinner("Fetching result"):
        try:
            html_to_display = _view_structure_from_alphafold_run(run_id=run_id, run_name=run_name, pdb_code=pdb_to_compare, include_pdb=include_pdb)
            components.html(html_to_display, height=700)
        except Exception as e:
            st.error(f"An error occured: {e}")


def render():
    user_info = get_user_info()
    st.markdown("###### Predict Protein Structure")

    view_model_choice = st.pills("Model:", ["ESMFold", "AlphaFold2", "Boltz"],
                                 selection_mode="single", default="ESMFold")

    if view_model_choice == "ESMFold":
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        with c1:
            view_esmfold_input_sequence = st.text_area("Provide an input sequence to infer the structure:",
                                                       "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE",
                                                       key="view_esmfold_input_sequence")
        with c2:
            view_structure_esmfold_btn = st.button("View", key="view_structure_esmfold_btn")
            clear_view_esmfold_btn = st.button("Clear", key="clear_view_esmfold_btn")

        prot_viewer = st.container()
        if view_structure_esmfold_btn:
            with st.spinner("Generating structure.."):
                with prot_viewer:
                    html = _esmfold_btn_fn(view_esmfold_input_sequence)
                    components.html(html, height=540)

        if clear_view_esmfold_btn:
            prot_viewer.empty()

    if view_model_choice == "AlphaFold2":
        st.write("Runs a workflow to do MSA and template search and then perform structure prediction")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            view_alphafold_input_sequence = st.text_area("Provide an input sequence to infer the structure:",
                                                         "QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS",
                                                         key="view_alphafold_input_sequence")

        c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with c1:
            view_alphafold_run_experiment = st.text_input("MLflow Experiment:", value="alphafold_structure_prediction")
        with c2:
            view_alphafold_run_label = st.text_input("Run Name:", value=f"alphafold_{datetime.now().strftime('%Y%m%d_%H%M')}")
        with c3:
            view_structure_alphafold_btn = st.button("Start Job", key="view_structure_alphafold_btn")

        if view_structure_alphafold_btn:
            is_valid = True
            if view_alphafold_input_sequence.strip() == "":
                is_valid = False
                st.error("Enter a valid sequence with the region to be replaced marked by square braces")

            if view_alphafold_run_experiment.strip() == "" or view_alphafold_run_label.strip() == "":
                is_valid = False
                st.error("Enter an mlflow experiment and run name")

            if is_valid:
                user_info = get_user_info()
                try:
                    with st.spinner("Starting job"):
                        alphafold_job_run_id = start_run_alphafold_job(
                            protein_sequence=view_alphafold_input_sequence,
                            mlflow_experiment_name=view_alphafold_run_experiment,
                            mlflow_run_name=view_alphafold_run_label,
                            user_info=user_info)

                        st.success(f"Job started with run id: {alphafold_job_run_id}.")
                        run_alphafold_job_id = os.getenv("RUN_ALPHAFOLD_JOB_ID")
                        st.button("View Run", on_click=lambda: open_run_window(run_alphafold_job_id, alphafold_job_run_id))

                except Exception as e:
                    st.error("An error occured while running the workflow")
                    print(e)

        st.divider()
        st.markdown("###### Search Past Runs:")
        c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")

        with c1:
            search_alphafold_run_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                                                 selection_mode="single", default="Experiment Name")
        with c2:
            search_alphafold_text = st.text_input(f"{search_alphafold_run_mode} contains:", "", key="search_alphafold_text")
        with c3:
            search_alphafold_run_button = st.button("Search", key="search_alphafold_run_button")

        if search_alphafold_run_button:
            with st.spinner("Searching"):
                if "alphafold_run_search_result_df" in st.session_state:
                    del st.session_state["alphafold_run_search_result_df"]
                    if "selected_alphafold_run_status" in st.session_state:
                        del st.session_state["selected_alphafold_run_status"]

                if search_alphafold_text.strip() != "":
                    if search_alphafold_run_mode == "Experiment Name":
                        alphafold_run_search_result_df = search_alphafold_runs_by_experiment_name(
                            user_email=user_info.user_email, experiment_name=search_alphafold_text)
                    else:
                        alphafold_run_search_result_df = search_alphafold_runs_by_run_name(
                            user_email=user_info.user_email, run_name=search_alphafold_text)

                    if not alphafold_run_search_result_df.empty:
                        st.session_state["alphafold_run_search_result_df"] = alphafold_run_search_result_df
                    else:
                        st.error("No results found")
                else:
                    st.error("Provide a search text")

        if "alphafold_run_search_result_df" in st.session_state:
            st.divider()
            view_af_result_enabled = ("selected_alphafold_run_status" in st.session_state
                                      and st.session_state["selected_alphafold_run_status"] == "fold_complete")

            view_c1, view_c2, view_c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
            with view_c1:
                alphafold_result_view_btn = st.button("View", disabled=not view_af_result_enabled)

            alphafold_results_selected_row = st.dataframe(
                st.session_state["alphafold_run_search_result_df"],
                column_config={"run_id": None},
                use_container_width=True,
                hide_index=True,
                on_select=_set_selected_row_status,
                selection_mode="single-row",
                key="alphafold_run_search_result_display_df")

            selected_row_for_view = alphafold_results_selected_row.selection.rows
            if len(selected_row_for_view) > 0 and alphafold_result_view_btn:
                _display_view_alphafold_result_dialog(selected_row_for_view)

    if view_model_choice == "Boltz":
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        with c1:
            view_boltz_input_sequence = st.text_area(
                "Provide an input sequence to predict the structure:",
                "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE",
                key="view_boltz_input_sequence")
        with c2:
            view_structure_boltz_btn = st.button("Predict", key="view_structure_boltz_btn")
            clear_view_boltz_btn = st.button("Clear", key="clear_view_boltz_btn")

        boltz_viewer = st.container()
        if view_structure_boltz_btn:
            with st.spinner("Predicting structure with Boltz..."):
                try:
                    pdb_result = hit_boltz(view_boltz_input_sequence)
                    with boltz_viewer:
                        html = molstar_html_multibody(pdb_result)
                        components.html(html, height=540)
                except Exception as e:
                    st.error(f"Error predicting structure: {e}")

        if clear_view_boltz_btn:
            boltz_viewer.empty()
