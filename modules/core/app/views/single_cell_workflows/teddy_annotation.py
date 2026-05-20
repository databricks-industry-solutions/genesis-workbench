"""Single Cell — TEDDY joint cell-type + disease annotation tab.

Both ontology heads are always shown (per the build decision: no model selector).
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.streamlit_helper import get_user_info
from utils.single_cell_analysis import (download_singlecell_markers_df, search_singlecell_runs)
from utils.teddy_tools import annotate_clusters


def render():
    st.markdown("##### TEDDY Annotation — Cell Type + Disease")
    st.markdown(
        "Embed each cell with **Merck TEDDY-G** (foundation model, Apache-2.0), then "
        "majority-vote both **cell type** and **disease** against a curated CELLxGENE "
        "reference index. Both labels come from one neighbor lookup."
    )

    user_info = get_user_info()

    if "singlecell_processing_runs_df" not in st.session_state:
        with st.spinner("Loading available runs..."):
            try:
                scanpy_runs = search_singlecell_runs(user_email=user_info.user_email, processing_mode="scanpy")
                rapids_runs = search_singlecell_runs(user_email=user_info.user_email, processing_mode="rapids-singlecell")
                runs_df = pd.concat([scanpy_runs, rapids_runs], ignore_index=True)
                st.session_state["singlecell_processing_runs_df"] = runs_df
            except Exception as e:
                st.error(f"Error loading runs: {e}")
                return

    runs_df = st.session_state.get("singlecell_processing_runs_df", pd.DataFrame())
    if runs_df.empty:
        st.info("No completed processing runs found. Run an analysis first in the Raw Single Cell Processing tab.")
        return

    runs_df = runs_df.sort_values("start_time", ascending=False).copy()
    runs_df["display_name"] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1,
    )

    c1, c2, c3 = st.columns([5, 2, 2], vertical_alignment="bottom")
    with c1:
        selected_display = st.selectbox(
            "Select a completed processing run:",
            list(runs_df["display_name"]),
            key="teddy_run_select",
        )
    with c2:
        cells_per_cluster = st.number_input(
            "Cells per cluster:", min_value=3, max_value=200, value=20, step=5, key="teddy_cpc"
        )
    with c3:
        k_neighbors = st.number_input(
            "Neighbors (k):", min_value=5, max_value=200, value=50, step=5, key="teddy_k"
        )

    run_id = dict(zip(runs_df["display_name"], runs_df["run_id"]))[selected_display]
    annotate_btn = st.button("Run TEDDY Annotation", type="primary", key="teddy_run_btn")

    cache_key = f"teddy_results_{run_id}"
    markers_cache_key = f"teddy_markers_{run_id}"
    cluster_col_key = f"teddy_cluster_col_{run_id}"

    if annotate_btn:
        with st.spinner("Loading data from MLflow..."):
            try:
                markers_df = download_singlecell_markers_df(run_id)
            except Exception as e:
                st.error(f"Failed to load run data: {e}")
                return

        cluster_col = None
        for c in ["cluster", "leiden", "louvain"]:
            if c in markers_df.columns:
                cluster_col = c
                break
        if not cluster_col:
            st.error("No cluster column found in the data.")
            return

        progress = st.progress(0, text="Starting TEDDY annotation...")
        spinner = st.empty()
        with spinner, st.spinner("Calling TEDDY endpoint + reference index..."):
            try:
                cluster_results_df = annotate_clusters(
                    markers_df,
                    cluster_col=cluster_col,
                    cells_per_cluster=cells_per_cluster,
                    k_neighbors=k_neighbors,
                    progress_callback=lambda pct, text: progress.progress(min(int(pct), 100), text=text),
                )
                st.session_state[cache_key] = cluster_results_df
                st.session_state[markers_cache_key] = markers_df
                st.session_state[cluster_col_key] = cluster_col
            except Exception as e:
                st.error(f"TEDDY annotation failed: {e}")
                return

    if cache_key in st.session_state:
        cluster_results_df = st.session_state[cache_key]
        markers_df = st.session_state[markers_cache_key]
        cluster_col = st.session_state[cluster_col_key]

        st.markdown("##### Per-Cluster Predictions")
        st.dataframe(cluster_results_df, use_container_width=True, hide_index=True)

        # Two UMAPs side-by-side: one colored by predicted cell type, one by predicted disease.
        if "UMAP_0" in markers_df.columns and "UMAP_1" in markers_df.columns:
            ct_map = dict(zip(cluster_results_df["Cluster"].astype(str), cluster_results_df["Predicted Cell Type"]))
            ds_map = dict(zip(cluster_results_df["Cluster"].astype(str), cluster_results_df["Predicted Disease"]))
            plot_df = markers_df.copy()
            plot_df["Predicted Cell Type"] = plot_df[cluster_col].astype(str).map(ct_map).fillna("Unknown")
            plot_df["Predicted Disease"] = plot_df[cluster_col].astype(str).map(ds_map).fillna("Unknown")

            left, right = st.columns(2)
            with left:
                fig_ct = px.scatter(
                    plot_df, x="UMAP_0", y="UMAP_1", color="Predicted Cell Type",
                    title="UMAP — TEDDY Cell Type", height=500, template="plotly_dark",
                )
                fig_ct.update_traces(marker=dict(size=3, opacity=0.7))
                fig_ct.update_layout(legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_ct, use_container_width=True)
            with right:
                fig_ds = px.scatter(
                    plot_df, x="UMAP_0", y="UMAP_1", color="Predicted Disease",
                    title="UMAP — TEDDY Disease", height=500, template="plotly_dark",
                )
                fig_ds.update_traces(marker=dict(size=3, opacity=0.7))
                fig_ds.update_layout(legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_ds, use_container_width=True)

