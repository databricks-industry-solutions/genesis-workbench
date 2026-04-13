"""Single Cell -- Gene Perturbation Prediction tab via scGPT."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.streamlit_helper import get_user_info, get_endpoint_name
from utils.single_cell_analysis import (download_singlecell_markers_df, search_singlecell_runs)
from databricks.sdk import WorkspaceClient


def _hit_perturbation_endpoint(expression, gene_names, genes_to_perturb, perturbation_type):
    """Call the scGPT perturbation endpoint."""
    endpoint_name = get_endpoint_name("scGPT Perturbation")
    ws = WorkspaceClient()

    model_input = [{
        "expression": expression,
        "gene_names": json.dumps(gene_names),
    }]
    response = ws.serving_endpoints.query(
        name=endpoint_name,
        inputs=model_input,
        params={
            "genes_to_perturb": ",".join(genes_to_perturb) if isinstance(genes_to_perturb, list) else genes_to_perturb,
            "perturbation_type": perturbation_type,
        },
    )
    return response.predictions


def render():
    st.markdown("##### Gene Perturbation Prediction")
    st.markdown(
        "Predict the effect of gene knockouts or overexpression on cell state "
        "using scGPT's transformer architecture. Select a completed processing run, "
        "pick a cluster, and specify genes to perturb."
    )

    user_info = get_user_info()

    # Check if endpoint is deployed
    try:
        get_endpoint_name("scGPT Perturbation")
    except RuntimeError:
        st.warning(
            "The scGPT Perturbation endpoint is not deployed yet. "
            "Deploy the scGPT module to enable perturbation prediction."
        )
        return

    # Load runs (reuse same filtered set as annotation tab)
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
        st.info("No completed processing runs found. Run an analysis first.")
        return

    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["display_name"] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1,
    )

    # Run selection
    col1, col2 = st.columns([5, 2], vertical_alignment="bottom")
    with col1:
        selected_display = st.selectbox("Select a completed processing run:", list(runs_df["display_name"]), key="perturb_run_select")
    with col2:
        load_btn = st.button("Load Run", key="perturb_load_btn", type="primary")

    run_id = dict(zip(runs_df["display_name"], runs_df["run_id"]))[selected_display]

    if load_btn:
        with st.spinner("Loading data from MLflow..."):
            try:
                markers_df = download_singlecell_markers_df(run_id)
                st.session_state["perturb_markers_df"] = markers_df
                st.session_state["perturb_run_id"] = run_id
            except Exception as e:
                st.error(f"Failed to load run: {e}")
                return

    if "perturb_markers_df" not in st.session_state:
        return

    markers_df = st.session_state["perturb_markers_df"]

    # Get cluster column
    cluster_col = None
    for c in ["cluster", "leiden", "louvain"]:
        if c in markers_df.columns:
            cluster_col = c
            break
    if not cluster_col:
        st.error("No cluster column found.")
        return

    clusters = sorted(markers_df[cluster_col].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    expr_cols = [c for c in markers_df.columns if c.startswith("expr_")]
    gene_names = [c.replace("expr_", "") for c in expr_cols]

    st.markdown("---")

    # Perturbation parameters — cluster selection
    col1, col2 = st.columns([2, 3], vertical_alignment="bottom")
    with col1:
        perturb_cluster = st.selectbox("Cluster:", clusters, key="perturb_cluster")
    with col2:
        perturb_type = st.radio("Perturbation Type:", ["Knockout", "Overexpress"], horizontal=True, key="perturb_type")

    # Rank genes by mean expression in the selected cluster for the selector
    cluster_cells = markers_df[markers_df[cluster_col] == perturb_cluster]
    mean_expr = cluster_cells[expr_cols].mean().sort_values(ascending=False)
    ranked_genes = [c.replace("expr_", "") for c in mean_expr.index]
    # Annotate with expression level for context
    gene_options = [f"{g} ({mean_expr[f'expr_{g}']:.2f})" for g in ranked_genes]
    gene_option_to_name = {opt: g for opt, g in zip(gene_options, ranked_genes)}

    st.markdown("**Select gene(s) to perturb:**")
    sel_col1, sel_col2 = st.columns([3, 2])
    with sel_col1:
        selected_gene_options = st.multiselect(
            "Choose from cluster marker genes (ranked by expression):",
            gene_options,
            default=[],
            key="perturb_gene_multiselect",
            help="Genes ranked by mean expression in the selected cluster. Expression level shown in parentheses.",
        )
    with sel_col2:
        extra_genes_input = st.text_input(
            "Or type additional gene(s):",
            placeholder="e.g., TP53, BRCA1",
            help="Comma-separated gene names not in the marker list above",
            key="perturb_extra_genes",
        )

    # Combine selected genes from both inputs
    genes_from_select = [gene_option_to_name[opt] for opt in selected_gene_options]
    genes_from_text = [g.strip().upper() for g in extra_genes_input.split(",") if g.strip()] if extra_genes_input else []
    all_genes_to_perturb = list(dict.fromkeys(genes_from_select + genes_from_text))  # deduplicate, preserve order

    perturb_genes_display = ", ".join(all_genes_to_perturb) if all_genes_to_perturb else ""

    predict_btn = st.button("Predict Perturbation Effect", type="primary", key="perturb_predict_btn")

    perturb_cache_key = f"perturb_{run_id}_{perturb_cluster}_{perturb_genes_display}_{perturb_type}"

    if predict_btn:
        if not all_genes_to_perturb:
            st.error("Please select or enter at least one gene to perturb.")
            return

        genes_to_perturb = all_genes_to_perturb

        # Get mean expression for the selected cluster
        cluster_cells = markers_df[markers_df[cluster_col] == perturb_cluster]
        mean_expression = cluster_cells[expr_cols].mean().values.tolist()

        status_container = st.container()
        with status_container:
            progress = st.progress(0, text="Preparing perturbation request...")
            spinner_placeholder = st.empty()

        with spinner_placeholder, st.spinner("Running perturbation prediction..."):
            try:
                progress.progress(30, text="Calling scGPT perturbation endpoint...")
                result = _hit_perturbation_endpoint(
                    expression=mean_expression,
                    gene_names=gene_names,
                    genes_to_perturb=genes_to_perturb,
                    perturbation_type=perturb_type.lower(),
                )
                progress.progress(90, text="Processing results...")

                if isinstance(result, list) and len(result) > 0:
                    result = result[0] if isinstance(result[0], dict) else result

                result_df = pd.DataFrame(result)
                result_df = result_df.sort_values("abs_delta", ascending=False)
                st.session_state[perturb_cache_key] = result_df
                progress.progress(100, text="Prediction complete!")
            except Exception as e:
                st.error(f"Perturbation prediction failed: {e}")
                return

    if perturb_cache_key in st.session_state:
        result_df = st.session_state[perturb_cache_key]
        st.markdown("---")
        st.markdown(f"##### Perturbation Results: {perturb_genes_input} ({perturb_type})")

        # Top affected genes bar chart
        top_n = min(20, len(result_df))
        top_genes = result_df.head(top_n)

        fig_bar = px.bar(
            top_genes,
            x="delta", y="gene_name",
            orientation="h",
            title=f"Top {top_n} Most Affected Genes",
            labels={"delta": "Expression Change (delta)", "gene_name": "Gene"},
            color="delta",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            height=max(400, top_n * 25),
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter plot: original vs predicted
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                result_df,
                x="original_expression", y="predicted_expression",
                hover_data=["gene_name", "delta"],
                title="Original vs Predicted Expression",
                labels={"original_expression": "Original", "predicted_expression": "Predicted"},
                height=400,
            )
            # Add y=x reference line
            max_val = max(result_df["original_expression"].max(), result_df["predicted_expression"].max())
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                  line=dict(dash="dash", color="gray"))
            fig_scatter.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.markdown("**Summary**")
            st.metric("Total genes analyzed", len(result_df))
            sig_genes = result_df[result_df["abs_delta"] > result_df["abs_delta"].quantile(0.95)]
            st.metric("Significantly affected (top 5%)", len(sig_genes))
            st.metric("Max |delta|", f"{result_df['abs_delta'].max():.4f}")

        # Full results table
        with st.expander("Full Results Table"):
            st.dataframe(
                result_df[["gene_name", "original_expression", "predicted_expression", "delta", "abs_delta"]].head(100),
                use_container_width=True, hide_index=True,
            )
