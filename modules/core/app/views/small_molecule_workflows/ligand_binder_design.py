"""Workflow 2: Small-Molecule Ligand Binder Design using Proteina-Complexa-Ligand + DiffDock validation."""

import streamlit as st
import streamlit.components.v1 as components
from utils.small_molecule_tools import (
    hit_proteina_complexa_ligand, hit_diffdock,
    molstar_html_pdb, molstar_html_protein_and_sdf,
    EXAMPLE_SMILES,
)

# A small ligand PDB (HETATM records) — 4-methoxybenzonitrile positioned near origin
EXAMPLE_LIGAND_PDB = """HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.394   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG A   1       2.091   1.209   0.000  1.00  0.00           C
HETATM    4  C4  LIG A   1       1.394   2.418   0.000  1.00  0.00           C
HETATM    5  C5  LIG A   1       0.000   2.418   0.000  1.00  0.00           C
HETATM    6  C6  LIG A   1      -0.697   1.209   0.000  1.00  0.00           C
HETATM    7  O1  LIG A   1       3.461   1.209   0.000  1.00  0.00           O
HETATM    8  C7  LIG A   1       4.158   2.418   0.000  1.00  0.00           C
HETATM    9  C8  LIG A   1      -2.115   1.209   0.000  1.00  0.00           C
HETATM   10  N1  LIG A   1      -3.277   1.209   0.000  1.00  0.00           N
END
"""

WORKFLOW_DESCRIPTION = (
    "Design a protein that binds a specific small molecule using "
    "[Proteina-Complexa-Ligand](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa). "
    "Validate the designed protein-ligand interaction using DiffDock molecular docking."
)


def render():
    st.markdown("### Ligand Binder Design")
    st.markdown(WORKFLOW_DESCRIPTION)

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        with st.form("ligand_binder_form", enter_to_submit=False):
            ligand_pdb = st.text_area("Ligand (PDB with HETATM records):", value=EXAMPLE_LIGAND_PDB, height=250,
                                      help="PDB content containing the small-molecule ligand as HETATM records")
            ligand_smiles = st.text_input("Ligand SMILES (for DiffDock validation):", value=EXAMPLE_SMILES,
                                          help="SMILES string of the same ligand, used for DiffDock docking validation")

            c1, c2 = st.columns(2)
            with c1:
                binder_len_min = st.number_input("Min protein length:", value=50, min_value=20, max_value=200)
            with c2:
                binder_len_max = st.number_input("Max protein length:", value=80, min_value=20, max_value=300)

            num_samples = st.slider("Number of designs:", min_value=1, max_value=10, value=2)
            validate_diffdock = st.checkbox("Validate with DiffDock", value=True,
                                            help="Run DiffDock on designed proteins to verify ligand binding")
            run_btn = st.form_submit_button("Design Ligand Binders", type="primary")

    with viewer_col:
        viewer_placeholder = st.empty()

        if "ligand_binder_results" in st.session_state and st.session_state["ligand_binder_results"] is not None:
            results_df = st.session_state["ligand_binder_results"]

            selected_idx = st.selectbox(
                "Select design:",
                options=results_df.index,
                format_func=lambda i: f"Design {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Design {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            # If DiffDock validation was done, show protein + docked ligand
            if "best_dock_sdf" in results_df.columns and row.get("best_dock_sdf"):
                with viewer_placeholder:
                    html = molstar_html_protein_and_sdf(row["pdb_output"], row["best_dock_sdf"])
                    components.html(html, height=540)
                st.caption(f"Showing: designed protein + DiffDock best pose (confidence: {row.get('dock_confidence', 'N/A'):.4f})")
            elif row.get("pdb_output"):
                with viewer_placeholder:
                    html = molstar_html_pdb(row["pdb_output"])
                    components.html(html, height=540)
                st.caption("Showing: designed protein (CA-only backbone)")

            st.markdown(f"**Designed Sequence:** `{row['sequence']}`")

            with st.expander("All designs"):
                display_cols = ["sample_id", "sequence", "rewards"]
                if "dock_confidence" in results_df.columns:
                    display_cols.append("dock_confidence")
                st.dataframe(
                    results_df[[c for c in display_cols if c in results_df.columns]],
                    use_container_width=True, hide_index=True,
                )

    if run_btn:
        if not ligand_pdb.strip():
            st.error("Ligand PDB (HETATM records) is required.")
            return

        with st.spinner("Running Proteina-Complexa-Ligand design..."):
            try:
                results_df = hit_proteina_complexa_ligand(
                    target_pdb=ligand_pdb,
                    binder_length_min=binder_len_min, binder_length_max=binder_len_max,
                    num_samples=num_samples,
                )
            except Exception as e:
                st.error(f"Ligand binder design failed: {e}")
                return

        if len(results_df) == 0:
            st.error("No designs returned.")
            return

        if validate_diffdock and ligand_smiles.strip():
            with st.spinner("Validating designs with DiffDock..."):
                dock_sdfs = []
                dock_scores = []
                for _, row in results_df.iterrows():
                    try:
                        dock_df = hit_diffdock(row["pdb_output"], ligand_smiles, samples_per_complex=5)
                        best = dock_df.sort_values("confidence", ascending=False).iloc[0]
                        dock_sdfs.append(best["ligand_sdf"])
                        dock_scores.append(best["confidence"])
                    except Exception:
                        dock_sdfs.append(None)
                        dock_scores.append(None)
                results_df["best_dock_sdf"] = dock_sdfs
                results_df["dock_confidence"] = dock_scores

        st.session_state["ligand_binder_results"] = results_df
        st.rerun()
