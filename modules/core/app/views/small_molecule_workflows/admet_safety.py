"""Workflow: ADMET & Safety profiling using ChemProp models."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_chemprop_bbbp, hit_chemprop_clintox, hit_chemprop_admet,
    EXAMPLE_SMILES,
)

WORKFLOW_DESCRIPTION = (
    "Profile small molecules for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties and safety using Chemprop D-MPNN models. "
    "Runs BBB penetration, clinical toxicity, and multi-task ADMET regression."
)

EXAMPLE_MOLECULES = """COc(cc1)ccc1C#N
CC(=O)Oc1ccccc1C(=O)O
CC(C)NCC(O)c1ccc(O)c(O)c1
C1CCCCC1
c1ccc2[nH]ccc2c1"""


def _risk_color(value, low_thresh=0.3, high_thresh=0.7):
    """Return color for risk indicator."""
    if value is None:
        return "gray"
    if value >= high_thresh:
        return "red"
    elif value >= low_thresh:
        return "orange"
    return "green"


def _risk_label(value, low_thresh=0.3, high_thresh=0.7):
    if value is None:
        return "N/A"
    if value >= high_thresh:
        return "High"
    elif value >= low_thresh:
        return "Medium"
    return "Low"


def render():
    st.markdown("### ADMET & Safety Profiling")
    st.markdown(WORKFLOW_DESCRIPTION)

    with st.form("admet_form", enter_to_submit=False):
        smiles_input = st.text_area("Enter SMILES (one per line):", value=EXAMPLE_MOLECULES, height=150,
                                     help="Enter one SMILES string per line. Each molecule will be profiled separately.")

        c1, c2, c3 = st.columns(3)
        with c1:
            run_bbbp = st.checkbox("BBB Penetration", value=True, help="Predict blood-brain barrier penetration probability")
        with c2:
            run_clintox = st.checkbox("Clinical Toxicity", value=True, help="Predict clinical trial toxicity probability")
        with c3:
            run_admet = st.checkbox("ADMET Properties", value=True, help="Predict 10 ADMET properties (absorption, metabolism, etc.)")

        st.markdown("**MLflow Tracking:**")
        mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_admet_safety", key="admet_mlflow_exp")
        mlflow_run_name = st.text_input("Run Name:", value="admet_profiling_run", key="admet_mlflow_run")
        run_btn = st.form_submit_button("Run ADMET Profiling", type="primary")

    # Display results
    if "admet_results" in st.session_state and st.session_state["admet_results"] is not None:
        results = st.session_state["admet_results"]

        for w in st.session_state.get("admet_warnings", []):
            st.warning(w)

        st.markdown("---")
        st.markdown("#### Results")

        # Summary cards
        smiles_list = results["smiles"]

        for idx, smi in enumerate(smiles_list):
            with st.expander(f"**{smi}**", expanded=(idx == 0)):
                cols = st.columns(3)

                # BBB
                with cols[0]:
                    if "bbbp" in results and results["bbbp"][idx] is not None:
                        bbbp_val = float(results["bbbp"][idx])
                        color = _risk_color(bbbp_val)
                        st.metric("BBB Penetration", f"{bbbp_val:.2%}")
                        st.caption(f":{color}[{'Permeable' if bbbp_val >= 0.5 else 'Non-permeable'}]")
                    else:
                        st.metric("BBB Penetration", "N/A")

                # Toxicity
                with cols[1]:
                    if "clintox" in results and results["clintox"][idx] is not None:
                        tox_val = float(results["clintox"][idx])
                        color = _risk_color(tox_val)
                        st.metric("Toxicity Risk", f"{tox_val:.2%}")
                        st.caption(f":{color}[{_risk_label(tox_val)}]")
                    else:
                        st.metric("Toxicity Risk", "N/A")

                # ADMET summary
                with cols[2]:
                    if "admet" in results and idx < len(results["admet_df"]):
                        admet_row = results["admet_df"].iloc[idx]
                        n_props = sum(1 for v in admet_row if pd.notna(v))
                        st.metric("ADMET Properties", f"{n_props} predicted")
                    else:
                        st.metric("ADMET Properties", "N/A")

                # ADMET detail table
                if "admet" in results and idx < len(results["admet_df"]):
                    st.markdown("**ADMET Properties:**")
                    admet_row = results["admet_df"].iloc[idx]
                    admet_display = pd.DataFrame({
                        "Property": admet_row.index,
                        "Value": admet_row.values
                    })
                    st.dataframe(admet_display, use_container_width=True, hide_index=True)

        # Full results table
        st.markdown("---")
        with st.expander("Full Results Table"):
            summary_data = {"SMILES": smiles_list}
            if "bbbp" in results:
                summary_data["BBB Penetration"] = [f"{v:.4f}" if v is not None else "N/A" for v in results["bbbp"]]
            if "clintox" in results:
                summary_data["Toxicity"] = [f"{v:.4f}" if v is not None else "N/A" for v in results["clintox"]]
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        if "admet_experiment_id" in st.session_state:
            st.button("View MLflow Experiment", key="admet_mlflow_btn",
                      on_click=lambda: open_mlflow_experiment_window(st.session_state["admet_experiment_id"]))

    # Execution
    if run_btn:
        if not smiles_input or not smiles_input.strip():
            st.error("Enter at least one SMILES string.")
            return
        if not mlflow_run_name or not mlflow_run_name.strip():
            st.error("MLflow Run Name is required.")
            return

        smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
        if not smiles_list:
            st.error("No valid SMILES found.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email)

        total_steps = sum([run_bbbp, run_clintox, run_admet])
        if total_steps == 0:
            st.error("Select at least one model to run.")
            return

        step = 0
        warnings = []
        results = {"smiles": smiles_list}

        progress = st.progress(0, text="Starting ADMET profiling...")

        with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
            mlflow.log_params({"num_molecules": len(smiles_list), "run_bbbp": run_bbbp,
                               "run_clintox": run_clintox, "run_admet": run_admet})

            if run_bbbp:
                progress.progress(int(step / total_steps * 100), text="Running BBB Penetration prediction...")
                try:
                    bbbp_preds = hit_chemprop_bbbp(smiles_list)
                    results["bbbp"] = bbbp_preds
                    mlflow.log_dict({"bbbp_predictions": bbbp_preds}, "bbbp_results.json")
                except Exception as e:
                    results["bbbp"] = [None] * len(smiles_list)
                    warnings.append(f"BBB Penetration prediction failed: {e}")
                step += 1

            if run_clintox:
                progress.progress(int(step / total_steps * 100), text="Running Clinical Toxicity prediction...")
                try:
                    clintox_preds = hit_chemprop_clintox(smiles_list)
                    results["clintox"] = clintox_preds
                    mlflow.log_dict({"clintox_predictions": clintox_preds}, "clintox_results.json")
                except Exception as e:
                    results["clintox"] = [None] * len(smiles_list)
                    warnings.append(f"Clinical Toxicity prediction failed: {e}")
                step += 1

            if run_admet:
                progress.progress(int(step / total_steps * 100), text="Running ADMET property prediction...")
                try:
                    admet_df = hit_chemprop_admet(smiles_list)
                    results["admet"] = True
                    results["admet_df"] = admet_df
                    mlflow.log_dict(admet_df.to_dict(), "admet_results.json")
                except Exception as e:
                    results["admet"] = False
                    results["admet_df"] = pd.DataFrame()
                    warnings.append(f"ADMET property prediction failed: {e}")
                step += 1

            progress.progress(100, text="Complete")
            st.session_state["admet_results"] = results
            st.session_state["admet_experiment_id"] = experiment.experiment_id
            st.session_state["admet_warnings"] = warnings
            mlflow.end_run(status="FINISHED")
        st.rerun()
