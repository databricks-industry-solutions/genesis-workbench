"""Single Cell -- Gene Perturbation Prediction tab via scGPT."""

import json
import traceback
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.streamlit_helper import get_user_info, get_endpoint_name
from utils.single_cell_analysis import (download_singlecell_markers_df, search_singlecell_runs)
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


def _hit_perturbation_endpoint(expression, gene_names, genes_to_perturb, perturbation_type):
    """Call the scGPT perturbation endpoint.

    Embeds the perturbation parameters (genes_to_perturb, perturbation_type)
    directly in the input payload rather than using the SDK params kwarg,
    which may not be passed correctly to custom PyFunc models.
    """
    endpoint_name = get_endpoint_name("scGPT Perturbation")
    ws = WorkspaceClient()

    # Pack everything into the input — the model's predict() will read
    # genes_to_perturb and perturbation_type from model_input if params is empty
    model_input = [{
        "expression": expression,
        "gene_names": json.dumps(gene_names),
        "genes_to_perturb": ",".join(genes_to_perturb) if isinstance(genes_to_perturb, list) else genes_to_perturb,
        "perturbation_type": perturbation_type,
    }]
    logger.info(f"Calling perturbation endpoint {endpoint_name} with {len(expression)} genes, perturbing {genes_to_perturb}")

    try:
        response = ws.serving_endpoints.query(
            name=endpoint_name,
            inputs=model_input,
        )
        logger.info(f"Perturbation endpoint responded")
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Perturbation endpoint call failed: {e}")
        raise RuntimeError(f"Endpoint call failed: {e}") from e

    if response.predictions is None:
        raise RuntimeError(f"Endpoint {endpoint_name} returned no predictions. Check endpoint logs for errors.")
    return response.predictions


def render():
    st.markdown("##### Gene Perturbation Prediction")
    st.markdown(
        "Predict the effect of gene knockouts or overexpression on cell state "
        "using scGPT's transformer architecture. Select a completed processing run, "
        "pick a cluster, and specify genes to perturb."
    )

    user_info = get_user_info()

    # Check if endpoint is actually deployed and reachable (not just in the map)
    try:
        endpoint_name = get_endpoint_name("scGPT Perturbation")
        ws = WorkspaceClient()
        ep_info = ws.serving_endpoints.get(name=endpoint_name)
        ep_state = ep_info.state.ready if ep_info.state else None
        if ep_state and ep_state.value == "NOT_READY":
            st.warning(
                f"The scGPT Perturbation endpoint (`{endpoint_name}`) exists but is not ready. "
                "It may be starting up or updating. Please wait and try again."
            )
    except RuntimeError:
        st.warning(
            "The scGPT Perturbation endpoint is not configured. "
            "Add it to `_MODEL_ENDPOINT_MAP` in streamlit_helper.py."
        )
        return
    except Exception as e:
        st.warning(
            f"The scGPT Perturbation endpoint (`{endpoint_name}`) is not deployed or not accessible: {e}\n\n"
            "Deploy the scGPT module with perturbation tasks enabled to use this feature."
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
                st.success(f"Loaded {len(markers_df):,} cells")
            except Exception as e:
                st.error(f"Failed to load run data: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
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
                progress.progress(20, text="Validating inputs...")
                if len(mean_expression) == 0:
                    st.error("No expression data found for the selected cluster.")
                    return
                if len(gene_names) != len(mean_expression):
                    st.error(f"Gene count ({len(gene_names)}) does not match expression count ({len(mean_expression)}).")
                    return

                progress.progress(30, text=f"Calling scGPT perturbation endpoint (perturbing {', '.join(genes_to_perturb)})...")
                result = _hit_perturbation_endpoint(
                    expression=mean_expression,
                    gene_names=gene_names,
                    genes_to_perturb=genes_to_perturb,
                    perturbation_type=perturb_type.lower(),
                )
                progress.progress(80, text="Processing results...")

                if isinstance(result, list) and len(result) > 0:
                    result = result[0] if isinstance(result[0], dict) else result

                if not isinstance(result, dict) or "gene_name" not in result:
                    st.error(f"Unexpected response format from endpoint. Got: {type(result)}")
                    with st.expander("Raw response"):
                        st.json(str(result)[:2000])
                    return

                result_df = pd.DataFrame(result)
                if result_df.empty:
                    st.warning("Prediction returned no results. The perturbed genes may not be in the model vocabulary.")
                    return

                # Ensure numeric columns
                for col in ["original_expression", "predicted_expression", "delta", "abs_delta"]:
                    if col in result_df.columns:
                        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

                result_df = result_df.sort_values("abs_delta", ascending=False)
                if result_df["abs_delta"].sum() == 0:
                    st.warning(
                        "All expression deltas are zero — the perturbation may not have been applied. "
                        "Check that the selected gene(s) exist in the scGPT model vocabulary."
                    )
                st.session_state[perturb_cache_key] = result_df
                progress.progress(100, text="Prediction complete")
            except Exception as e:
                st.error(f"Perturbation prediction failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                return

    if perturb_cache_key in st.session_state:
        result_df = st.session_state[perturb_cache_key]
        st.markdown("---")
        st.markdown(f"##### Perturbation Results: {perturb_genes_display} ({perturb_type})")

        try:
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
                template="plotly_dark",
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
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
                    template="plotly_dark",
                )
                max_val = max(result_df["original_expression"].max(), result_df["predicted_expression"].max())
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                      line=dict(dash="dash", color="gray"))
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                st.markdown("**Summary**")
                st.metric("Total genes analyzed", len(result_df))
                sig_genes = result_df[result_df["abs_delta"] > result_df["abs_delta"].quantile(0.95)]
                st.metric("Significantly affected (top 5%)", len(sig_genes))
                st.metric("Max |delta|", f"{result_df['abs_delta'].max():.4f}")

            # Full results table
            with st.expander("Full Results Table"):
                display_df = result_df[["gene_name", "original_expression", "predicted_expression", "delta", "abs_delta"]].head(100).copy()
                # Ensure all numeric columns are float to avoid Arrow serialization issues
                for col in ["original_expression", "predicted_expression", "delta", "abs_delta"]:
                    display_df[col] = display_df[col].astype(float)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error displaying results: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            with st.expander("Raw result data"):
                st.write(result_df.dtypes.to_dict())
                st.write(result_df.head(5).to_dict())
