import streamlit as st
import streamlit.components.v1 as components
import os

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models)

from utils.streamlit_helper import (get_user_info,
                                    display_deploy_model_dialog,
                                    display_import_model_uc_dialog)

from utils.small_molecule_tools import (hit_diffdock,
                                        molstar_html_protein_and_sdf,
                                        EXAMPLE_PDB,
                                        EXAMPLE_SMILES)

from views.small_molecule_workflows import binder_design, ligand_binder_design, motif_scaffolding

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
        deployed_small_molecule_models_df = get_deployed_models(ModelCategory.SMALL_MOLECULES)
        deployed_small_molecule_models_df.columns = ["Model Id", "Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        st.session_state["deployed_small_molecule_models_df"] = deployed_small_molecule_models_df
    deployed_small_molecule_models_df = st.session_state["deployed_small_molecule_models_df"]

user_info = get_user_info()

settings_tab, diffdock_tab, binder_tab, ligand_binder_tab, motif_tab = st.tabs([
    "Settings",
    "Molecular Docking",
    "Protein Binder Design",
    "Ligand Binder Design",
    "Motif Scaffolding",
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
                             "Endpoint Name": None
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
            protein_pdb = st.text_area("Target Protein (PDB):", value=EXAMPLE_PDB, height=300,
                                       help="Paste PDB content for the target protein")
            num_samples = st.slider("Number of poses:", min_value=1, max_value=20, value=5,
                                    help="Number of docking poses to generate")
            run_docking = st.form_submit_button("Run Docking", type="primary")

    with viewer_col:
        viewer_placeholder = st.empty()

        if "diffdock_results" in st.session_state and st.session_state["diffdock_results"] is not None:
            results_df = st.session_state["diffdock_results"]
            protein_pdb_stored = st.session_state.get("diffdock_protein_pdb", "")

            selected_rank = st.selectbox(
                "Select pose:",
                options=results_df.index,
                format_func=lambda i: f"Rank {results_df.loc[i, 'rank']} — Confidence: {results_df.loc[i, 'confidence']:.4f}"
            )

            ligand_sdf = results_df.loc[selected_rank, "ligand_sdf"]
            if not ligand_sdf.startswith("ERROR"):
                html = molstar_html_protein_and_sdf(protein_pdb_stored, ligand_sdf)
                with viewer_placeholder:
                    components.html(html, height=540)
            else:
                st.error(f"Pose generation failed: {ligand_sdf}")

            with st.expander("All docking results"):
                st.dataframe(
                    results_df[["rank", "confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )

    if run_docking:
        if not protein_pdb.strip() or not ligand_smiles.strip():
            st.error("Both protein PDB and molecule SMILES are required.")
        else:
            with st.spinner("Running DiffDock molecular docking..."):
                try:
                    results_df = hit_diffdock(protein_pdb, ligand_smiles, num_samples)
                    if len(results_df) > 0:
                        st.session_state["diffdock_results"] = results_df
                        st.session_state["diffdock_protein_pdb"] = protein_pdb
                        st.rerun()
                    else:
                        st.error("DiffDock returned no results.")
                except Exception as e:
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
