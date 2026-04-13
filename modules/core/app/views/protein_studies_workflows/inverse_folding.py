"""Protein Studies -- Inverse Folding tab: ProteinMPNN standalone."""

import streamlit as st
import streamlit.components.v1 as components

from utils.molstar_tools import molstar_html_multibody
from utils.protein_design import hit_proteinmpnn, hit_esmfold

# A small, valid PDB backbone (ubiquitin fragment, 10 residues, chain A)
_DEFAULT_PDB = """\
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  N   GLN A   2      26.335  27.770   3.258  1.00  9.27           N
ATOM      6  CA  GLN A   2      26.850  29.021   3.898  1.00 10.07           C
ATOM      7  C   GLN A   2      26.100  29.253   5.202  1.00  9.68           C
ATOM      8  O   GLN A   2      24.865  29.024   5.330  1.00  9.38           O
ATOM      9  N   ILE A   3      26.849  29.569   6.250  1.00 10.00           N
ATOM     10  CA  ILE A   3      26.235  30.050   7.497  1.00 10.25           C
ATOM     11  C   ILE A   3      26.882  31.410   7.862  1.00 10.97           C
ATOM     12  O   ILE A   3      28.032  31.634   7.483  1.00 12.41           O
ATOM     13  N   PHE A   4      26.106  32.305   8.468  1.00  9.85           N
ATOM     14  CA  PHE A   4      26.574  33.660   8.776  1.00 10.62           C
ATOM     15  C   PHE A   4      26.644  34.482   7.486  1.00 10.33           C
ATOM     16  O   PHE A   4      25.724  34.471   6.669  1.00 10.89           O
ATOM     17  N   VAL A   5      27.741  35.183   7.310  1.00 10.10           N
ATOM     18  CA  VAL A   5      27.956  35.969   6.083  1.00 10.68           C
ATOM     19  C   VAL A   5      28.406  37.363   6.480  1.00 11.22           C
ATOM     20  O   VAL A   5      29.024  37.528   7.535  1.00 12.64           O
ATOM     21  N   LYS A   6      28.068  38.349   5.643  1.00 11.00           N
ATOM     22  CA  LYS A   6      28.399  39.737   5.953  1.00 12.16           C
ATOM     23  C   LYS A   6      27.170  40.466   6.518  1.00 12.50           C
ATOM     24  O   LYS A   6      26.022  40.004   6.313  1.00 12.23           O
ATOM     25  N   THR A   7      27.439  41.518   7.285  1.00 12.76           N
ATOM     26  CA  THR A   7      26.340  42.374   7.741  1.00 13.04           C
ATOM     27  C   THR A   7      26.934  43.770   7.965  1.00 13.42           C
ATOM     28  O   THR A   7      28.140  43.956   8.110  1.00 14.32           O
ATOM     29  N   LEU A   8      26.046  44.758   8.043  1.00 13.16           N
ATOM     30  CA  LEU A   8      26.451  46.148   8.258  1.00 14.51           C
ATOM     31  C   LEU A   8      25.695  46.713   9.457  1.00 14.39           C
ATOM     32  O   LEU A   8      24.462  46.672   9.483  1.00 15.33           O
END
"""


def render():
    st.markdown("###### Inverse Folding with ProteinMPNN")
    st.markdown("Given a protein backbone structure (PDB), design new amino acid sequences that would fold into that structure.")

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        inv_fold_pdb_input = st.text_area(
            "Paste PDB content (backbone structure):",
            value=_DEFAULT_PDB,
            height=200,
            key="inv_fold_pdb_input",
        )
    with c2:
        inv_fold_btn = st.button("Design Sequences", key="inv_fold_btn", type="primary")
        inv_fold_clear = st.button("Clear", key="inv_fold_clear_btn")

    if inv_fold_btn:
        if not inv_fold_pdb_input.strip() or not inv_fold_pdb_input.strip().startswith("ATOM"):
            st.error("Please paste a valid PDB structure starting with ATOM records")
        else:
            with st.spinner("Running ProteinMPNN..."):
                try:
                    sequences = hit_proteinmpnn(inv_fold_pdb_input)
                    st.session_state["inv_fold_sequences"] = sequences
                    st.session_state["inv_fold_last_validated_idx"] = None
                    st.session_state.pop("inv_fold_validated_pdb", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running ProteinMPNN: {e}")

    if inv_fold_clear:
        for key in ["inv_fold_sequences", "inv_fold_validated_pdb", "inv_fold_last_validated_idx"]:
            st.session_state.pop(key, None)
        st.rerun()

    if "inv_fold_sequences" in st.session_state:
        sequences = st.session_state["inv_fold_sequences"]
        st.markdown("---")

        # Design selector
        design_options = [f"Design {i + 1}" for i in range(len(sequences))]
        selected_design = st.selectbox("Select design:", design_options, key="inv_fold_design_select")
        selected_idx = design_options.index(selected_design)

        # Show selected sequence
        st.markdown(f"**{selected_design} -- Sequence:**")
        st.code(sequences[selected_idx], language=None)

        # Auto-fold: fold whenever the selection changes
        last_validated = st.session_state.get("inv_fold_last_validated_idx")
        if last_validated != selected_idx:
            with st.spinner(f"Folding {selected_design} with ESMFold..."):
                try:
                    pdb_result = hit_esmfold(sequences[selected_idx])
                    st.session_state["inv_fold_validated_pdb"] = pdb_result
                    st.session_state["inv_fold_last_validated_idx"] = selected_idx
                except Exception as e:
                    st.error(f"ESMFold failed: {e}")
                    st.session_state.pop("inv_fold_validated_pdb", None)
                    st.session_state["inv_fold_last_validated_idx"] = selected_idx

        if "inv_fold_validated_pdb" in st.session_state:
            st.markdown("**Predicted Structure:**")
            html = molstar_html_multibody(st.session_state["inv_fold_validated_pdb"])
            components.html(html, height=540)
