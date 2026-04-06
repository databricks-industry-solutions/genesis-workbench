"""Workflow 3: Functional Motif Scaffolding using Proteina-Complexa-AME + ProteinMPNN + ESMFold."""

import streamlit as st
import streamlit.components.v1 as components
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_proteina_complexa_ame, hit_esmfold,
    molstar_html_pdb, molstar_html_multi_pdb,
    EXAMPLE_PDB,
)

# Example motif PDB — a small active-site fragment with ligand context
EXAMPLE_MOTIF_PDB = """ATOM      1  N   HIS B   1       5.123   8.456   2.345  1.00 15.00           N
ATOM      2  CA  HIS B   1       5.891   7.234   2.789  1.00 15.00           C
ATOM      3  C   HIS B   1       7.321   7.567   3.123  1.00 15.00           C
ATOM      4  O   HIS B   1       7.654   8.678   3.567  1.00 15.00           O
ATOM      5  CB  HIS B   1       5.456   6.123   3.678  1.00 15.00           C
ATOM      6  CG  HIS B   1       4.012   5.789   3.456  1.00 15.00           C
ATOM      7  ND1 HIS B   1       3.123   6.567   4.123  1.00 15.00           N
ATOM      8  CE1 HIS B   1       1.890   6.012   3.890  1.00 15.00           C
ATOM      9  NE2 HIS B   1       1.987   4.890   3.123  1.00 15.00           N
ATOM     10  CD2 HIS B   1       3.234   4.678   2.890  1.00 15.00           C
ATOM     11  N   ASP B   2       8.123   6.567   2.890  1.00 15.00           N
ATOM     12  CA  ASP B   2       9.543   6.789   3.234  1.00 15.00           C
ATOM     13  C   ASP B   2      10.234   5.567   3.890  1.00 15.00           C
ATOM     14  O   ASP B   2       9.678   4.456   4.012  1.00 15.00           O
ATOM     15  CB  ASP B   2      10.123   7.890   2.345  1.00 15.00           C
ATOM     16  CG  ASP B   2      11.567   8.123   2.678  1.00 15.00           C
ATOM     17  OD1 ASP B   2      12.234   7.234   3.123  1.00 15.00           O
ATOM     18  OD2 ASP B   2      11.890   9.234   2.345  1.00 15.00           O
ATOM     19  N   SER B   3      11.456   5.678   4.234  1.00 15.00           N
ATOM     20  CA  SER B   3      12.234   4.567   4.890  1.00 15.00           C
ATOM     21  C   SER B   3      13.678   4.890   5.234  1.00 15.00           C
ATOM     22  O   SER B   3      14.123   5.987   5.012  1.00 15.00           O
ATOM     23  CB  SER B   3      11.890   3.234   4.234  1.00 15.00           C
ATOM     24  OG  SER B   3      12.567   2.123   4.678  1.00 15.00           O
HETATM   25  C1  LIG B   1       6.500   3.200   5.100  1.00  5.00           C
HETATM   26  O1  LIG B   1       7.200   2.100   5.500  1.00  5.00           O
HETATM   27  N1  LIG B   1       5.300   3.500   5.800  1.00  5.00           N
END
"""

WORKFLOW_DESCRIPTION = (
    "Transplant a functional motif (e.g., an enzyme active site or binding loop) into a new, "
    "stable protein scaffold using [Proteina-Complexa-AME](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa). "
    "Optionally optimize the scaffold sequence with ProteinMPNN and validate folding with ESMFold."
)


def _hit_proteinmpnn(pdb_str: str) -> list:
    """Call ProteinMPNN endpoint."""
    from utils.streamlit_helper import get_endpoint_name
    from utils.small_molecule_tools import _query_endpoint
    endpoint_name = get_endpoint_name("ProteinMPNN")
    result = _query_endpoint(endpoint_name, {"inputs": [pdb_str]})
    return result.get("predictions", result)


def render():
    st.markdown("### Functional Motif Scaffolding")
    st.markdown(WORKFLOW_DESCRIPTION)

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        with st.form("motif_scaffold_form", enter_to_submit=False):
            motif_pdb = st.text_area("Motif + Ligand (PDB):", value=EXAMPLE_MOTIF_PDB, height=250,
                                     help="PDB containing the functional motif residues (ATOM) and optional ligand (HETATM)")
            target_chain = st.text_input("Motif Chain:", value="B",
                                         help="Chain ID containing the motif residues")

            c1, c2 = st.columns(2)
            with c1:
                scaffold_len_min = st.number_input("Min scaffold length:", value=50, min_value=20, max_value=200)
            with c2:
                scaffold_len_max = st.number_input("Max scaffold length:", value=80, min_value=20, max_value=300)

            num_samples = st.slider("Number of scaffolds:", min_value=1, max_value=10, value=2)
            optimize_mpnn = st.checkbox("Optimize with ProteinMPNN", value=False,
                                        help="Run ProteinMPNN to optimize the scaffold sequence while preserving motif geometry")
            validate_esmfold = st.checkbox("Validate with ESMFold", value=True,
                                           help="Fold the designed sequence with ESMFold to verify structural integrity")

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_motif_scaffolding", key="motif_mlflow_exp")
            mlflow_run_name = st.text_input("Run Name:", key="motif_mlflow_run")
            run_btn = st.form_submit_button("Generate Scaffolds", type="primary")

    with viewer_col:
        status_container = st.container()
        viewer_placeholder = st.empty()

        if "scaffold_results" in st.session_state and st.session_state["scaffold_results"] is not None:
            results_df = st.session_state["scaffold_results"]

            selected_idx = st.selectbox(
                "Select scaffold:",
                options=results_df.index,
                key="motif_scaffold_selector",
                format_func=lambda i: f"Scaffold {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Scaffold {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            # Prefer ESMFold structure > MPNN > raw scaffold
            display_pdb = None
            caption = ""
            if "esmfold_pdb" in results_df.columns and row.get("esmfold_pdb"):
                display_pdb = row["esmfold_pdb"]
                caption = "ESMFold-validated scaffold"
            elif row.get("pdb_output"):
                display_pdb = row["pdb_output"]
                caption = "Designed scaffold (CA-only backbone)"

            if display_pdb:
                # Overlay with original motif
                motif_pdb_stored = st.session_state.get("scaffold_motif_pdb", "")
                with viewer_placeholder:
                    html = molstar_html_multi_pdb([motif_pdb_stored, display_pdb])
                    components.html(html, height=540)
                st.caption(f"Showing: original motif + {caption}")

            seq = row.get("mpnn_sequence", row["sequence"])
            st.markdown(f"**Sequence:** `{seq}`")

            if "scaffold_experiment_id" in st.session_state:
                st.button("View MLflow Experiment", key="scaffold_mlflow_btn",
                          on_click=lambda: open_mlflow_experiment_window(st.session_state["scaffold_experiment_id"]))

            with st.expander("All scaffolds"):
                display_cols = ["sample_id", "sequence", "rewards"]
                if "mpnn_sequence" in results_df.columns:
                    display_cols.append("mpnn_sequence")
                if "esmfold_validated" in results_df.columns:
                    display_cols.append("esmfold_validated")
                st.dataframe(
                    results_df[[c for c in display_cols if c in results_df.columns]],
                    use_container_width=True, hide_index=True,
                )

    if run_btn:
        if not motif_pdb.strip():
            st.error("Motif PDB is required.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email,
                                           host=None, token=None, shared=True)

        total_steps = 1 + (1 if optimize_mpnn else 0) + (1 if validate_esmfold else 0)
        step = 0

        with status_container:
            progress = st.progress(0, text="Generating scaffolds with Proteina-Complexa-AME...")
            spinner = st.empty()
        with spinner, st.spinner("Running.."):
          with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
            mlflow.log_params({"target_chain": target_chain, "scaffold_len_min": scaffold_len_min,
                               "scaffold_len_max": scaffold_len_max, "num_samples": num_samples,
                               "optimize_mpnn": optimize_mpnn, "validate_esmfold": validate_esmfold})

            try:
                results_df = hit_proteina_complexa_ame(
                    target_pdb=motif_pdb, target_chain=target_chain,
                    binder_length_min=scaffold_len_min, binder_length_max=scaffold_len_max,
                    num_samples=num_samples,
                )
            except Exception as e:
                st.error(f"Motif scaffolding failed: {e}")
                return

            if len(results_df) == 0:
                st.error("No scaffolds returned.")
                return

            mlflow.log_dict(results_df.to_dict(), "proteina_complexa_ame_results.json")
            step += 1
            progress.progress(int(step / total_steps * 100), text="Scaffolds generated")

            if optimize_mpnn:
                mpnn_seqs = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    base_pct = int(step / total_steps * 100)
                    next_pct = int((step + 1) / total_steps * 100)
                    pct = base_pct + int((idx + 1) / total * (next_pct - base_pct))
                    progress.progress(pct, text=f"Optimizing sequence {idx+1}/{total} with ProteinMPNN...")
                    try:
                        seqs = _hit_proteinmpnn(row["pdb_output"])
                        mpnn_seqs.append(seqs[0] if isinstance(seqs, list) and seqs else row["sequence"])
                    except Exception:
                        mpnn_seqs.append(row["sequence"])
                results_df["mpnn_sequence"] = mpnn_seqs
                mlflow.log_dict({"mpnn_sequences": mpnn_seqs}, "proteinmpnn_results.json")
                step += 1

            if validate_esmfold:
                esmfold_pdbs = []
                validated = []
                seq_col = "mpnn_sequence" if "mpnn_sequence" in results_df.columns else "sequence"
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    base_pct = int(step / total_steps * 100)
                    next_pct = int((step + 1) / total_steps * 100)
                    pct = base_pct + int((idx + 1) / total * (next_pct - base_pct))
                    progress.progress(pct, text=f"Validating scaffold {idx+1}/{total} with ESMFold...")
                    try:
                        pdb = hit_esmfold(row[seq_col])
                        esmfold_pdbs.append(pdb)
                        validated.append(True)
                    except Exception:
                        esmfold_pdbs.append(None)
                        validated.append(False)
                results_df["esmfold_pdb"] = esmfold_pdbs
                results_df["esmfold_validated"] = validated
                mlflow.log_dict({"esmfold_validated": validated}, "esmfold_validation.json")

            progress.progress(100, text="Complete")
            st.session_state["scaffold_results"] = results_df
            st.session_state["scaffold_motif_pdb"] = motif_pdb
            st.session_state["scaffold_experiment_id"] = experiment.experiment_id
            st.rerun()
