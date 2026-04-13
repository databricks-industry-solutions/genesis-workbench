"""Single Cell — Cell Similarity Search tab via SCimilarity endpoints."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.streamlit_helper import get_user_info
from utils.single_cell_analysis import (download_singlecell_markers_df, search_singlecell_runs)
from utils.scimilarity_tools import (get_gene_order, align_to_gene_order, lognorm_counts,
                                      get_cell_embeddings, search_nearest_cells)


def render():
    st.markdown("##### Cell Similarity Search")
    st.markdown("Search SCimilarity's 23M-cell reference database for cells similar to a selected cluster.")

    user_info = get_user_info()

    if "singlecell_runs_df" not in st.session_state:
        with st.spinner("Loading available runs..."):
            try:
                runs_df = search_singlecell_runs(user_email=user_info.user_email)
                st.session_state["singlecell_runs_df"] = runs_df
            except Exception as e:
                st.error(f"Error loading runs: {e}")
                return

    runs_df = st.session_state.get("singlecell_runs_df", pd.DataFrame())
    if runs_df.empty:
        st.info("No completed processing runs found. Run an analysis first.")
        return

    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["display_name"] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1,
    )

    col1, col2 = st.columns([5, 2], vertical_alignment="bottom")
    with col1:
        selected_display = st.selectbox("Select a completed processing run:", list(runs_df["display_name"]), key="similarity_run_select")
    with col2:
        load_for_sim = st.button("Load Run", key="sim_load_btn", type="primary")

    sim_run_id = dict(zip(runs_df["display_name"], runs_df["run_id"]))[selected_display]

    if load_for_sim:
        with st.spinner("Loading data from MLflow..."):
            try:
                markers_df = download_singlecell_markers_df(sim_run_id)
                st.session_state["sim_markers_df"] = markers_df
                st.session_state["sim_run_id"] = sim_run_id
            except Exception as e:
                st.error(f"Failed to load run: {e}")
                return

    if "sim_markers_df" not in st.session_state:
        return

    markers_df = st.session_state["sim_markers_df"]

    cluster_col = None
    for c in ["cluster", "leiden", "louvain"]:
        if c in markers_df.columns:
            cluster_col = c
            break
    if not cluster_col:
        st.error("No cluster column found.")
        return

    clusters = sorted(markers_df[cluster_col].unique(), key=lambda x: int(x) if str(x).isdigit() else x)

    col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    with col1:
        sim_cluster = st.selectbox("Query Cluster:", clusters, key="sim_cluster")
    with col2:
        sim_k = st.number_input("Neighbors (k):", min_value=10, max_value=1000, value=100, step=50, key="sim_k")
    with col3:
        sim_search_btn = st.button("Search", key="sim_search_btn", type="primary")

    sim_cache_key = f"sim_results_{sim_run_id}_{sim_cluster}_{sim_k}"

    if sim_search_btn:
        progress = st.progress(0, text="Starting similarity search...")
        try:
            progress.progress(10, text="Fetching gene order...")
            gene_order = get_gene_order()

            progress.progress(20, text="Preparing expression matrix...")
            expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
            gene_names = [c.replace("expr_", "") for c in expr_cols]
            expr_df = markers_df[expr_cols].copy()
            expr_df.columns = gene_names
            aligned = align_to_gene_order(expr_df, gene_order)
            normed = lognorm_counts(aligned)

            cl_mask = markers_df[cluster_col] == sim_cluster
            cl_indices = markers_df.index[cl_mask].tolist()
            n_sample = min(20, len(cl_indices))
            sampled = cl_indices[:n_sample]

            progress.progress(40, text=f"Generating embeddings for {n_sample} cells...")
            sample_normed = normed.loc[sampled]
            expression_json = sample_normed.to_json(orient="split")
            embeddings_result = get_cell_embeddings(expression_json)

            progress.progress(60, text="Searching reference database...")
            all_metadata = []
            total = len(embeddings_result)
            for i, row in embeddings_result.iterrows():
                embedding = row["embedding"]
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                pct = 60 + int((i + 1) / total * 30)
                progress.progress(pct, text=f"Searching cell {i + 1}/{total}...")
                try:
                    result = search_nearest_cells(embedding, k=sim_k)
                    meta_json = result.get("results_metadata") if isinstance(result, dict) else None
                    if meta_json and isinstance(meta_json, str):
                        nn_meta = pd.read_json(meta_json, orient="split") if "columns" in meta_json else pd.read_json(meta_json)
                    elif isinstance(meta_json, (dict, list)):
                        nn_meta = pd.DataFrame(meta_json)
                    else:
                        nn_meta = pd.DataFrame()
                    all_metadata.append(nn_meta)
                except Exception:
                    pass

            progress.progress(95, text="Aggregating results...")
            if all_metadata:
                combined = pd.concat(all_metadata, ignore_index=True)
                st.session_state[sim_cache_key] = combined
            else:
                st.warning("No search results returned.")
                return
            progress.progress(100, text="Search complete!")
        except Exception as e:
            st.error(f"Similarity search failed: {e}")
            return

    if sim_cache_key in st.session_state:
        combined = st.session_state[sim_cache_key]
        st.markdown("---")
        st.markdown(f"##### Results: Neighbors of Cluster {sim_cluster}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if "prediction" in combined.columns:
                type_counts = combined["prediction"].value_counts().head(15)
                fig_types = px.bar(x=type_counts.values, y=type_counts.index, orientation="h",
                    title="Neighbor Cell Types", labels={"x": "Count", "y": "Cell Type"},
                    color=type_counts.values, color_continuous_scale="Viridis")
                fig_types.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", showlegend=False, height=400)
                st.plotly_chart(fig_types, use_container_width=True)

        with chart_col2:
            if "disease" in combined.columns:
                disease_counts = combined["disease"].value_counts().head(15)
                fig_disease = px.bar(x=disease_counts.values, y=disease_counts.index, orientation="h",
                    title="Neighbor Disease Distribution", labels={"x": "Count", "y": "Disease"},
                    color=disease_counts.values, color_continuous_scale="Reds")
                fig_disease.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", showlegend=False, height=400)
                st.plotly_chart(fig_disease, use_container_width=True)

        if "study" in combined.columns:
            with st.expander("Neighbor Study Sources"):
                study_counts = combined["study"].value_counts().reset_index()
                study_counts.columns = ["Study", "Count"]
                st.dataframe(study_counts, use_container_width=True, hide_index=True)

        with st.expander("Full Results Table"):
            display_cols = [c for c in ["prediction", "disease", "study"] if c in combined.columns]
            st.dataframe(combined[display_cols].head(200) if display_cols else combined.head(200), use_container_width=True, hide_index=True)
