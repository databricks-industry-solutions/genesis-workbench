import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models,
                                      get_batch_models)

import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import (get_user_info,
                                    display_deploy_model_dialog,
                                    open_mlflow_experiment_window,
                                    display_import_model_uc_dialog)

from utils.small_molecule_tools import (hit_diffdock,
                                        molstar_html_multi_pdb,
                                        _sdf_to_hetatm,
                                        _get_example_pdb,
                                        EXAMPLE_SMILES)

from views.small_molecule_workflows import binder_design, ligand_binder_design, motif_scaffolding, admet_safety

st.title(":material/science: Small Molecules")

with st.spinner("Loading data"):
    if "available_small_molecule_models_df" not in st.session_state:
        available_small_molecule_models_df = get_available_models(ModelCategory.SMALL_MOLECULES)
        available_small_molecule_models_df["model_labels"] = (available_small_molecule_models_df["model_id"].astype(str) + " - "
                                            + available_small_molecule_models_df["model_display_name"].astype(str) + " [ "
                                            + available_small_molecule_models_df["model_uc_name"].astype(str) + " v"
                                            + available_small_molecule_models_df["model_uc_version"].astype(str) + " ]"
                                            )
        st.session_state["available_small_molecule_models_df"] = available_small_molecule_models_df
    available_small_molecule_models_df = st.session_state["available_small_molecule_models_df"]

    if "deployed_small_molecule_models_df" not in st.session_state:
        rt_df = get_deployed_models(ModelCategory.SMALL_MOLECULES)
        rt_df.columns = ["Model Id", "Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        rt_df["Type"] = "Real-time"
        rt_df["Cluster"] = ""
        rows = [rt_df]
        try:
            batch_df = get_batch_models("small_molecules")
            if not batch_df.empty:
                batch_df.columns = ["Name", "Description", "Endpoint Name", "Cluster"]
                batch_df["Type"] = "Batch"
                batch_df["Model Id"] = ""
                batch_df["Deploy Id"] = ""
                batch_df["Model Name"] = ""
                batch_df["Source Version"] = ""
                batch_df["UC Name/Version"] = ""
                rows.append(batch_df)
        except Exception:
            pass
        deployed_small_molecule_models_df = pd.concat(rows, ignore_index=True)
        st.session_state["deployed_small_molecule_models_df"] = deployed_small_molecule_models_df
    deployed_small_molecule_models_df = st.session_state["deployed_small_molecule_models_df"]

user_info = get_user_info()

settings_tab, diffdock_tab, binder_tab, ligand_binder_tab, motif_tab, admet_tab = st.tabs([
    "Deployed Models",
    "Molecular Docking",
    "Protein Binder Design",
    "Ligand Binder Design",
    "Motif Scaffolding",
    "ADMET & Safety",
])

# ── Settings Tab ──
with settings_tab:
    st.markdown("###### Available Models:")
    with st.form("deploy_model_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_model_for_deploy = st.selectbox("Model:", available_small_molecule_models_df["model_labels"], label_visibility="collapsed")
        with col2:
            deploy_button = st.form_submit_button('Deploy')
    if deploy_button:
        display_deploy_model_dialog(selected_model_for_deploy)

    if len(deployed_small_molecule_models_df) > 0:
        with st.form("modify_deployed_model_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("###### Deployed Models")
            with col2:
                st.form_submit_button("Manage")

            st.dataframe(deployed_small_molecule_models_df,
                         use_container_width=True,
                         hide_index=True,
                         on_select="rerun",
                         selection_mode="single-row",
                         column_config={
                             "Model Id": None,
                             "Deploy Id": None,
                             "Endpoint Name": None,
                             "Model Name": None,
                             "Source Version": None,
                         })
    else:
        st.write("There are no deployed models")

# ── Molecular Docking Tab (DiffDock) ──
with diffdock_tab:
    st.markdown("### Molecular Docking with DiffDock")
    st.markdown("Predict 3D binding poses for protein-ligand complexes using [DiffDock](https://github.com/gcorso/DiffDock) — "
                "a diffusion generative model that generates and ranks candidate docking poses with a confidence score.")

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        with st.form("diffdock_form", enter_to_submit=False):
            ligand_smiles = st.text_input("Molecule (SMILES):", value=EXAMPLE_SMILES,
                                          help="SMILES string for the small molecule ligand")
            protein_pdb = st.text_area("Target Protein (PDB):", value=_get_example_pdb(), height=300,
                                       help="Paste PDB content for the target protein")
            num_samples = st.slider("Number of poses:", min_value=1, max_value=20, value=5,
                                    help="Number of docking poses to generate")

            st.markdown("**MLflow Tracking:**")
            diffdock_mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_molecular_docking", key="diffdock_mlflow_exp")
            diffdock_mlflow_run_name = st.text_input("Run Name:", value="molecular_docking_run", key="diffdock_mlflow_run")
            run_docking = st.form_submit_button("Run Docking", type="primary")

    with viewer_col:
        status_container = st.container()

        if "diffdock_results" in st.session_state and st.session_state["diffdock_results"] is not None:
            results_df = st.session_state["diffdock_results"]
            protein_pdb_stored = st.session_state.get("diffdock_protein_pdb", "")

            selected_rank = st.selectbox(
                "Select pose:",
                options=results_df.index,
                format_func=lambda i: f"Rank {results_df.loc[i, 'rank']} — Confidence: {results_df.loc[i, 'confidence']:.4f}"
            )

            # Viewer inline — below selector, above controls
            ligand_sdf = results_df.loc[selected_rank, "ligand_sdf"]
            if not ligand_sdf.startswith("ERROR"):
                lig_hetatm = _sdf_to_hetatm(ligand_sdf)
                lig_pdb = lig_hetatm + "\nEND\n" if lig_hetatm else ""
                pdbs = [protein_pdb_stored] + ([lig_pdb] if lig_pdb else [])
                html = molstar_html_multi_pdb(pdbs)
                components.html(html, height=540)
            else:
                st.error(f"Pose generation failed: {ligand_sdf}")

            if "diffdock_experiment_id" in st.session_state:
                st.button("View MLflow Experiment", key="diffdock_mlflow_btn",
                          on_click=lambda: open_mlflow_experiment_window(st.session_state["diffdock_experiment_id"]))

            with st.expander("All docking results"):
                st.dataframe(
                    results_df[["rank", "confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )

    if run_docking:
        if not protein_pdb.strip() or not ligand_smiles.strip():
            st.error("Both protein PDB and molecule SMILES are required.")
        elif not diffdock_mlflow_run_name or not diffdock_mlflow_run_name.strip():
            st.error("MLflow Run Name is required.")
        else:
            user_info = get_user_info()
            experiment = set_mlflow_experiment(experiment_tag=diffdock_mlflow_experiment, user_email=user_info.user_email)

            with status_container:
                progress = st.progress(0, text="Preparing DiffDock run...")
            with mlflow.start_run(run_name=diffdock_mlflow_run_name, experiment_id=experiment.experiment_id) as run:
                mlflow.log_params({"ligand_smiles": ligand_smiles, "num_samples": num_samples})
                try:
                    from utils.small_molecule_tools import _query_endpoint, get_endpoint_name
                    import json as _json

                    # Step 1: ESM embeddings
                    progress.progress(10, text="Step 1/3: Computing ESM embeddings...")
                    esm_endpoint = get_endpoint_name("DiffDock ESM Embeddings")
                    esm_result = _query_endpoint(esm_endpoint, {
                        "dataframe_split": {
                            "columns": ["protein_pdb"],
                            "data": [[protein_pdb]]
                        }
                    })
                    esm_predictions = esm_result.get("predictions", esm_result)
                    if isinstance(esm_predictions, list):
                        embeddings_b64 = esm_predictions[0].get("embeddings_b64", "{}")
                    elif isinstance(esm_predictions, dict):
                        embeddings_b64 = esm_predictions.get("embeddings_b64", "{}")
                    else:
                        embeddings_b64 = "{}"

                    # Step 2: DiffDock scoring
                    progress.progress(40, text="Step 2/3: Running DiffDock pose generation...")
                    scoring_endpoint = get_endpoint_name("DiffDock")
                    result = _query_endpoint(scoring_endpoint, {
                        "dataframe_split": {
                            "columns": ["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
                            "data": [[protein_pdb, ligand_smiles, num_samples, embeddings_b64]]
                        }
                    })
                    results_df = pd.DataFrame(result.get("predictions", result))

                    # Step 3: Process results
                    progress.progress(80, text="Step 3/3: Processing results...")
                    if len(results_df) > 0:
                        # Check if any poses succeeded (not ERROR)
                        valid = results_df[~results_df["ligand_sdf"].str.startswith("ERROR", na=False)]
                        mlflow.log_dict(results_df[["rank", "confidence"]].to_dict(), "diffdock_results.json")
                        st.session_state["diffdock_results"] = results_df
                        st.session_state["diffdock_protein_pdb"] = protein_pdb
                        st.session_state["diffdock_experiment_id"] = experiment.experiment_id
                        if len(valid) == 0:
                            progress.progress(100, text="Completed with errors — all poses failed.")
                        else:
                            progress.progress(100, text=f"Complete — {len(valid)} pose(s) generated successfully.")
                        mlflow.end_run(status="FINISHED")
                        st.rerun()
                    else:
                        progress.progress(100, text="No results returned.")
                        st.error("DiffDock returned no results.")
                except Exception as e:
                    progress.progress(100, text="Failed.")
                    st.error(f"DiffDock inference failed: {str(e)}")

# ── Protein Binder Design Tab ──
with binder_tab:
    binder_design.render()

# ── Ligand Binder Design Tab ──
with ligand_binder_tab:
    ligand_binder_design.render()

# ── Motif Scaffolding Tab ──
with motif_tab:
    motif_scaffolding.render()

# ── ADMET & Safety Tab ──
with admet_tab:
    admet_safety.render()
