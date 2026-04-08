"""Protein Studies — Sequence Search tab: Hybrid funnel BLAST-like search."""

import streamlit as st
import pandas as pd
import logging

from utils.sequence_search_tools import run_sequence_search
from utils.streamlit_helper import get_user_info

logger = logging.getLogger(__name__)

EXAMPLE_SEQUENCE = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


def _get_progress_callback(progress_widget):
    """Create a progress callback for the search pipeline."""
    def report_progress(progress_pct, status_text):
        progress_widget.progress(progress_pct, text=status_text)
    return report_progress


def render():
    user_info = get_user_info()
    st.markdown("###### Sequence Similarity Search")
    st.caption("BLAST-like search using ESM-2 embeddings + Vector Search + Smith-Waterman alignment")

    # Input section
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        input_mode = st.radio("Input:", ["Paste Sequence", "Upload FASTA"],
                              horizontal=True, key="seq_search_input_mode")

        if input_mode == "Paste Sequence":
            query_sequence = st.text_area(
                "Enter amino acid sequence:",
                EXAMPLE_SEQUENCE,
                height=120,
                key="seq_search_input_sequence",
            )
        else:
            uploaded = st.file_uploader("Upload FASTA:", type=["fasta", "fa", "faa"],
                                        key="seq_search_file_upload")
            query_sequence = ""
            if uploaded:
                from Bio import SeqIO
                from io import StringIO
                content = uploaded.read().decode()
                record = next(SeqIO.parse(StringIO(content), "fasta"))
                query_sequence = str(record.seq)
                st.code(f">{record.id}\n{query_sequence[:80]}...", language=None)

    with c2:
        top_k = st.select_slider("Max Results:", options=[25, 50, 100, 200, 500],
                                  value=50, key="seq_search_top_k")
        search_btn = st.button("Search", type="primary", key="seq_search_btn")

    # Search execution
    if search_btn:
        clean_seq = query_sequence.strip().replace("\n", "").replace(" ", "")

        if not clean_seq:
            st.error("Please enter a valid amino acid sequence.")
            return

        # Validate sequence characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid_chars = set(clean_seq.upper()) - valid_aa
        if invalid_chars:
            st.warning(f"Sequence contains non-standard characters: {invalid_chars}. "
                       "Results may be affected.")

        progress_widget = st.progress(0, text="Starting search...")
        progress_callback = _get_progress_callback(progress_widget)

        try:
            results = run_sequence_search(
                query_sequence=clean_seq,
                top_k=top_k,
                progress_callback=progress_callback,
            )
            st.session_state["seq_search_results"] = results
            st.session_state["seq_search_query"] = clean_seq
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            st.error(f"Search failed: {e}")
            return

    # Results display
    if "seq_search_results" in st.session_state:
        results = st.session_state["seq_search_results"]

        if not results:
            st.info("No results found.")
            return

        st.divider()
        st.markdown(f"###### Results ({len(results)} hits)")

        # Build results dataframe
        results_data = [{
            "Seq ID": r.seq_id,
            "Description": r.description[:80] if r.description else "",
            "Identity %": r.identity_pct,
            "SW Score": r.sw_score,
            "Alignment Length": r.alignment_length,
            "Seq Length": r.seq_length,
            "Vector Distance": round(r.vector_distance, 4),
        } for r in results]

        results_df = pd.DataFrame(results_data)

        # Selectable dataframe
        selection = st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="seq_search_results_table",
        )

        # Alignment viewer for selected row
        selected_rows = selection.selection.rows if selection.selection else []
        if selected_rows:
            selected_idx = selected_rows[0]
            hit = results[selected_idx]

            st.divider()
            st.markdown(f"###### Alignment: {hit.seq_id}")
            st.caption(hit.description)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Identity", f"{hit.identity_pct}%")
            with col2:
                st.metric("SW Score", hit.sw_score)
            with col3:
                st.metric("Alignment Length", hit.alignment_length)

            # Show alignment
            # Display in chunks of 60 characters for readability
            chunk_size = 60
            alignment_lines = []
            for i in range(0, len(hit.aligned_query), chunk_size):
                q_chunk = hit.aligned_query[i:i + chunk_size]
                c_chunk = hit.aligned_comp[i:i + chunk_size]
                t_chunk = hit.aligned_target[i:i + chunk_size]
                alignment_lines.append(f"Query:  {q_chunk}")
                alignment_lines.append(f"        {c_chunk}")
                alignment_lines.append(f"Target: {t_chunk}")
                alignment_lines.append("")

            st.code("\n".join(alignment_lines), language=None)

            # Link to structure prediction
            if st.button("Predict structure for this hit (ESMFold)",
                         key="seq_search_predict_structure"):
                st.session_state["view_esmfold_input_sequence"] = hit.aligned_target.replace("-", "")
                st.info("Sequence copied. Switch to the 'Protein Structure Prediction' tab to run ESMFold.")
