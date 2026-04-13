"""Protein Studies — Sequence Search tab: Hybrid funnel BLAST-like search."""

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import logging

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from utils.sequence_search_tools import run_sequence_search
from utils.streamlit_helper import get_user_info
from utils.small_molecule_tools import hit_esmfold
from utils.molstar_tools import molstar_html_multibody

logger = logging.getLogger(__name__)


def _extract_organism(description: str) -> str:
    """Use Claude Sonnet to extract organism name from a UniRef sequence description."""
    endpoint = os.environ.get("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-6")
    try:
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=endpoint,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="Extract the organism name from the protein sequence description. "
                            "Return ONLY the organism name (e.g. 'Homo sapiens', 'Escherichia coli'). "
                            "If you cannot determine the organism, return 'Unknown'."
                ),
                ChatMessage(role=ChatMessageRole.USER, content=description),
            ],
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Unknown"

EXAMPLE_SEQUENCE = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


@st.dialog("Sequence Match Details", width="large")
def _display_hit_dialog(selected_idx):
    """Dialog showing alignment details (left) and ESMFold structure (right)."""
    results_sorted = st.session_state.get("seq_search_results_sorted", [])
    if selected_idx >= len(results_sorted):
        st.error("Invalid selection.")
        return

    hit = results_sorted[selected_idx]

    st.markdown(f"##### {hit.seq_id}")
    st.caption(hit.description)

    # Extract organism name using LLM
    with st.spinner("Identifying organism..."):
        organism = _extract_organism(hit.description) if hit.description else "Unknown"
    st.markdown(f"**Suggested Organism:** {organism}")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Identity", f"{hit.identity_pct}%")
    with mc2:
        st.metric("SW Score", hit.sw_score)
    with mc3:
        st.metric("Alignment Length", hit.alignment_length)
    with mc4:
        st.metric("Seq Length", hit.seq_length)

    with st.expander("Sequence Alignment", expanded=True):
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

    st.markdown("**Predicted Structure**")
    target_seq = hit.aligned_target.replace("-", "")
    with st.spinner("Predicting structure with ESMFold..."):
        try:
            pdb_str = hit_esmfold(target_seq)
            html = molstar_html_multibody(pdb_str)
            components.html(html, height=540)
        except Exception as e:
            st.warning(f"Structure prediction unavailable: {e}")


def _get_progress_callback(progress_widget):
    """Create a progress callback for the search pipeline."""
    def report_progress(progress_pct, status_text):
        progress_widget.progress(progress_pct, text=status_text)
    return report_progress


@st.dialog("How Sequence Search Works", width="large")
def _show_search_info():
    st.markdown("""
This search finds similar protein sequences across **~150M UniRef90 entries** in under 5 seconds using a 5-stage hybrid funnel:

| Stage | What happens | Time |
|-------|-------------|------|
| **1. Embed** | Your query is converted to a 1280-dimensional vector using ESM-2 | ~200ms |
| **2. ANN Search** | Vector Search finds the top 500 nearest neighbors by embedding similarity | ~500ms |
| **3. Fetch** | Full sequences are retrieved from the Delta table for each candidate | ~1s |
| **4. Align** | Smith-Waterman alignment scores each candidate against your query | ~2s |
| **5. Rank** | Results are sorted by alignment score and returned | instant |

**Key difference from BLAST:** Stage 1-2 use *semantic* similarity (ESM-2 captures evolutionary and functional relationships), while Stage 4 provides *exact* alignment scores. This combination catches functionally similar sequences that BLAST might miss, while still providing familiar identity % and alignment views.
""")


def render():
    user_info = get_user_info()
    st.markdown("###### Sequence Similarity Search")
    cap_c1, cap_c2, _ = st.columns([8, 2,10])
    with cap_c1:
        st.caption("BLAST-like search using ESM-2 embeddings + Vector Search + Smith-Waterman alignment")
        if st.button(":material/info: More Info", key="seq_search_info_btn", help="How does this search work?"):
            _show_search_info()

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
        st.caption("Sorted by best match (highest sequence identity first)")

        # Build results dataframe, sorted by identity descending
        results_sorted = sorted(results, key=lambda r: r.identity_pct, reverse=True)
        st.session_state["seq_search_results_sorted"] = results_sorted

        results_data = [{
            "Seq ID": r.seq_id,
            "Description": r.description[:80] if r.description else "",
            "Identity %": r.identity_pct,
            "SW Score": r.sw_score,
            "Alignment Length": r.alignment_length,
            "Seq Length": r.seq_length,
            "Vector Distance": round(r.vector_distance, 4),
        } for r in results_sorted]

        results_df = pd.DataFrame(results_data)
        st.session_state["seq_search_results_df"] = results_df

        # View Results button + selectable dataframe
        def _set_selected_seq_row():
            sel = st.session_state["seq_search_results_table"].selection
            if len(sel["rows"]) > 0:
                st.session_state["seq_search_selected_idx"] = sel["rows"][0]
            else:
                st.session_state.pop("seq_search_selected_idx", None)

        view_btn = st.button("View Results", key="seq_search_view_btn",
                             disabled="seq_search_selected_idx" not in st.session_state)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select=_set_selected_seq_row,
            key="seq_search_results_table",
        )

        if view_btn and "seq_search_selected_idx" in st.session_state:
            _display_hit_dialog(st.session_state["seq_search_selected_idx"])
