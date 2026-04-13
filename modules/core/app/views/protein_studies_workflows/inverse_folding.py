"""Protein Studies — Inverse Folding tab: ProteinMPNN standalone."""

import streamlit as st
import streamlit.components.v1 as components

from utils.molstar_tools import molstar_html_multibody
from utils.protein_design import hit_proteinmpnn, hit_esmfold


def render():
    st.markdown("###### Inverse Folding with ProteinMPNN")
    st.markdown("Given a protein backbone structure (PDB), design new amino acid sequences that would fold into that structure.")

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        inv_fold_pdb_input = st.text_area(
            "Paste PDB content (backbone structure):",
            height=200,
            placeholder="ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  9.67           N\n...",
            key="inv_fold_pdb_input",
        )
    with c2:
        inv_fold_btn = st.button("Design Sequences", key="inv_fold_btn", type="primary")
        inv_fold_validate = st.checkbox("Validate with ESMFold", value=False, key="inv_fold_validate",
                                        help="Fold each designed sequence with ESMFold and display the structure")
        inv_fold_clear = st.button("Clear", key="inv_fold_clear_btn")

    inv_fold_viewer = st.container()

    if inv_fold_btn:
        if not inv_fold_pdb_input.strip() or not inv_fold_pdb_input.strip().startswith("ATOM"):
            st.error("Please paste a valid PDB structure starting with ATOM records")
        else:
            with st.spinner("Running ProteinMPNN..."):
                try:
                    sequences = hit_proteinmpnn(inv_fold_pdb_input)
                    st.session_state["inv_fold_sequences"] = sequences
                except Exception as e:
                    st.error(f"Error running ProteinMPNN: {e}")

    if "inv_fold_sequences" in st.session_state:
        sequences = st.session_state["inv_fold_sequences"]
        with inv_fold_viewer:
            st.markdown(f"**{len(sequences)} designed sequences:**")
            for i, seq in enumerate(sequences):
                st.code(seq, language=None)

            if inv_fold_validate and sequences:
                st.markdown("---")
                st.markdown("##### ESMFold Validation")
                for i, seq in enumerate(sequences):
                    with st.spinner(f"Folding sequence {i + 1}/{len(sequences)} with ESMFold..."):
                        try:
                            pdb_result = hit_esmfold(seq)
                            st.markdown(f"**Design {i + 1}**")
                            html = molstar_html_multibody(pdb_result)
                            components.html(html, height=500)
                        except Exception as e:
                            st.warning(f"ESMFold failed for sequence {i + 1}: {e}")

    if inv_fold_clear:
        if "inv_fold_sequences" in st.session_state:
            del st.session_state["inv_fold_sequences"]
        inv_fold_viewer.empty()
