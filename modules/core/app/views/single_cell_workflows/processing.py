"""Single Cell — Raw Single Cell Processing tab: run analysis and view results."""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from mlflow.tracking import MlflowClient

from utils.streamlit_helper import get_user_info
from utils.single_cell_analysis import (start_scanpy_job,
                                        start_rapids_singlecell_job,
                                        download_singlecell_markers_df,
                                        download_cluster_markers_mapping,
                                        get_mlflow_run_url,
                                        search_singlecell_runs)


def _display_run_analysis():
    st.markdown("###### Run Analysis")

    st.markdown("**Analysis Mode:**")
    mode_display = st.selectbox(
        "Mode",
        options=["scanpy", "rapids-singlecell [GPU-accelerated]"],
        label_visibility="collapsed",
        key="scanpy_mode_selector"
    )
    mode = "rapids-singlecell" if "rapids-singlecell" in mode_display else mode_display
    default_experiment = "rapidssinglecell_genesis_workbench" if mode == "rapids-singlecell" else "scanpy_genesis_workbench"
    default_run_name = "rapidssinglecell_analysis" if mode == "rapids-singlecell" else "scanpy_analysis"

    with st.form("scanpy_analysis_form", enter_to_submit=False):
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Data Configuration:**")
            data_path = st.text_input("Data Path (h5ad file)", placeholder="/Volumes/catalog/schema/volume/file.h5ad",
                                      help="Path to the h5ad file in Unity Catalog Volumes")
            gene_name_column = st.text_input("Gene Name Column (optional)", value="", placeholder="e.g., gene_name, feature_name",
                                             help="Name of column in var containing gene names. Leave empty to use Ensembl reference mapping.")

            if not gene_name_column or gene_name_column.strip() == "":
                species = st.selectbox("Species", options=["hsapiens", "mmusculus", "rnorvegicus"], index=0,
                                       help="Species for Ensembl gene name mapping.",
                                       format_func=lambda x: {"hsapiens": "Human (Homo sapiens)", "mmusculus": "Mouse (Mus musculus)", "rnorvegicus": "Rat (Rattus norvegicus)"}[x])
            else:
                species = "hsapiens"

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input("MLflow Experiment Name", value=default_experiment)
            mlflow_run_name = st.text_input("MLflow Run Name", value=default_run_name)

        with col2:
            st.markdown("**Filtering Parameters:**")
            min_genes = st.number_input("Min Genes per Cell", min_value=0, value=200, step=10)
            min_cells = st.number_input("Min Cells per Gene", min_value=0, value=3, step=1)
            pct_counts_mt = st.number_input("Max % Mitochondrial Counts", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            n_genes_by_counts = st.number_input("Max Genes by Counts", min_value=0, value=2500, step=100)

        st.divider()
        col3, col4 = st.columns([1, 1])
        with col3:
            st.markdown("**Normalization & Feature Selection:**")
            target_sum = st.number_input("Target Sum for Normalization", min_value=0, value=10000, step=1000)
            n_top_genes = st.number_input("Number of Highly Variable Genes", min_value=0, value=500, step=50)
        with col4:
            st.markdown("**Dimensionality Reduction & Clustering:**")
            n_pcs = st.number_input("Number of Principal Components", min_value=0, value=50, step=5)
            cluster_resolution = st.number_input("Cluster Resolution", min_value=0.0, max_value=2.0, value=0.15, step=0.05, format="%.2f")

        st.divider()
        submit_button = st.form_submit_button("Start Analysis", use_container_width=True)

    if submit_button:
        is_valid = True
        if not data_path.strip():
            st.error("Please provide a data path"); is_valid = False
        elif not data_path.endswith(".h5ad"):
            st.error("Data path must point to an .h5ad file"); is_valid = False
        elif not data_path.startswith("/Volumes"):
            st.error("Data path must start with /Volumes (Unity Catalog Volume)"); is_valid = False
        if not mlflow_experiment.strip() or not mlflow_run_name.strip():
            st.error("Please provide MLflow experiment and run names"); is_valid = False
        if (not gene_name_column or gene_name_column.strip() == "") and (not species or species == ""):
            st.error("Please either provide a Gene Name Column OR select a Species for reference mapping."); is_valid = False

        if is_valid:
            user_info = get_user_info()
            params = dict(data_path=data_path, mlflow_experiment=mlflow_experiment, mlflow_run_name=mlflow_run_name,
                          gene_name_column=gene_name_column, species=species, min_genes=min_genes, min_cells=min_cells,
                          pct_counts_mt=pct_counts_mt, n_genes_by_counts=n_genes_by_counts, target_sum=target_sum,
                          n_top_genes=n_top_genes, n_pcs=n_pcs, cluster_resolution=cluster_resolution, user_info=user_info)

            start_fn = start_scanpy_job if mode == "scanpy" else start_rapids_singlecell_job if mode == "rapids-singlecell" else None
            if start_fn is None:
                st.error(f"Unknown mode: {mode}"); return

            try:
                with st.spinner(f"Starting {mode} analysis job..."):
                    job_id, job_run_id = start_fn(**params)
                    host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                    if not host_name.startswith("https://"):
                        host_name = "https://" + host_name
                    run_url = f"{host_name}/jobs/{job_id}/runs/{job_run_id}"
                    st.success(f"Job started successfully! Run ID: {job_run_id}")
                    st.link_button("View Run in Databricks", run_url, type="primary")
            except Exception as e:
                st.error(f"An error occurred while starting the job: {str(e)}")
                print(e)


def _get_cluster_column(df):
    if 'cluster' in df.columns: return 'cluster'
    elif 'leiden' in df.columns: return 'leiden'
    elif 'louvain' in df.columns: return 'louvain'
    return None


def _display_results_viewer():
    st.markdown("##### Single-Cell Results Viewer")
    st.markdown("Select a completed single-cell analysis run to visualize")

    user_info = get_user_info()

    if 'date_filter' not in st.session_state:
        st.session_state['date_filter'] = None
    if 'processing_mode_filter' not in st.session_state:
        st.session_state['processing_mode_filter'] = "All"

    main_col1, main_col2, main_col3, main_col4 = st.columns([5, 1.5, 2.5, 1], vertical_alignment="bottom")

    with main_col1:
        experiment_filter = st.text_input("MLflow Experiment:", value="genesis_workbench",
                                          help="Enter experiment name to filter runs (partial match supported).")

    with main_col2:
        processing_mode_option = st.radio("Mode:", ["All", "Scanpy", "Rapids-SingleCell"],
                                          index=["All", "Scanpy", "Rapids-SingleCell"].index(st.session_state['processing_mode_filter']),
                                          help="Filter by processing pipeline")
        if processing_mode_option != st.session_state['processing_mode_filter']:
            st.session_state['processing_mode_filter'] = processing_mode_option
            if 'singlecell_runs_df' in st.session_state:
                del st.session_state['singlecell_runs_df']

    with main_col3:
        st.markdown("**Time Period:**")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if st.button("Today", use_container_width=True):
                st.session_state['date_filter'] = 0
                if 'singlecell_runs_df' in st.session_state: del st.session_state['singlecell_runs_df']
        with row1_col2:
            if st.button("Last 7 Days", use_container_width=True):
                st.session_state['date_filter'] = 7
                if 'singlecell_runs_df' in st.session_state: del st.session_state['singlecell_runs_df']
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if st.button("Last 30 Days", use_container_width=True):
                st.session_state['date_filter'] = 30
                if 'singlecell_runs_df' in st.session_state: del st.session_state['singlecell_runs_df']
        with row2_col2:
            if st.button("All Time", use_container_width=True):
                st.session_state['date_filter'] = None
                if 'singlecell_runs_df' in st.session_state: del st.session_state['singlecell_runs_df']

    with main_col4:
        st.write(""); st.write("")
        refresh_button = st.button("Refresh", use_container_width=True)

    if refresh_button or 'singlecell_runs_df' not in st.session_state:
        with st.spinner("Searching for your single-cell analysis runs..."):
            try:
                processing_mode = None
                if st.session_state['processing_mode_filter'] != "All":
                    processing_mode = st.session_state['processing_mode_filter'].lower().replace("-", "-")
                    if processing_mode == "rapids-singlecell": processing_mode = "rapids-singlecell"
                    elif processing_mode == "scanpy": processing_mode = "scanpy"
                runs_df = search_singlecell_runs(user_email=user_info.user_email, processing_mode=processing_mode, days_back=st.session_state['date_filter'])
                st.session_state['singlecell_runs_df'] = runs_df
                if len(runs_df) == 0: st.info("No single-cell analysis runs found. Run an analysis first!")
            except Exception as e:
                st.error(f"Error searching runs: {str(e)}"); return

    runs_df = st.session_state.get('singlecell_runs_df', pd.DataFrame())
    if len(runs_df) == 0:
        st.info("No single-cell runs found. Go to 'Run New Analysis' to create one!"); return

    if experiment_filter and experiment_filter.strip():
        runs_df = runs_df[runs_df['experiment'].str.contains(experiment_filter.strip(), case=False, na=False)]
        if len(runs_df) == 0:
            st.warning(f"No runs found matching experiment: '{experiment_filter}'"); return

    st.divider()
    runs_df = runs_df.sort_values('start_time', ascending=False)
    runs_df['display_name'] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}", axis=1)
    run_options = dict(zip(runs_df['display_name'], runs_df['run_id']))

    select_col1, select_col2, select_col3 = st.columns([4, 5, 1], vertical_alignment="bottom")
    with select_col1:
        run_name_filter = st.text_input("Search by Run Name:", value="", placeholder="Type to filter runs...")

    if run_name_filter and run_name_filter.strip():
        filtered_runs = runs_df[runs_df['run_name'].str.contains(run_name_filter.strip(), case=False, na=False)]
        if len(filtered_runs) == 0:
            st.warning(f"No runs found matching run name: '{run_name_filter}'"); return
        filtered_options = dict(zip(filtered_runs['display_name'], filtered_runs['run_id']))
    else:
        filtered_runs = runs_df
        filtered_options = run_options

    with select_col2:
        st.markdown(f"**{len(filtered_runs)} Runs:**")
        selected_display = st.selectbox("Select:", list(filtered_options.keys()), label_visibility="collapsed")
    with select_col3:
        st.write("")
        load_button = st.button("Load", type="primary", use_container_width=True)

    run_id = filtered_options[selected_display]

    if load_button:
        with st.spinner("Loading data from MLflow..."):
            try:
                df = download_singlecell_markers_df(run_id)
                mlflow_url = get_mlflow_run_url(run_id)
                st.session_state['singlecell_df'] = df
                st.session_state['singlecell_run_id'] = run_id
                st.session_state['singlecell_mlflow_url'] = mlflow_url
                st.success(f"Loaded {len(df):,} cells with {len(df.columns)} features")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Tip: Make sure this run includes the markers_flat.parquet artifact")
                return

    if 'singlecell_df' not in st.session_state:
        return

    df = st.session_state['singlecell_df']
    run_id = st.session_state.get('singlecell_run_id', '')
    mlflow_url = st.session_state.get('singlecell_mlflow_url', '')

    st.markdown("---")
    st.markdown("##### Interactive UMAP Visualization")
    st.info("This viewer displays a subsampled dataset (max 10,000 cells) with marker genes only for faster, interactive plotting. "
            "The complete output AnnData object with all genes is available in the MLflow run.")

    expr_cols = [c for c in df.columns if c.startswith('expr_')]
    obs_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    obs_numerical = [c for c in df.select_dtypes(include=['number']).columns.tolist() if not c.startswith(('UMAP_', 'PC_'))]

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        color_type = st.selectbox("Color by:", ["Cluster", "Marker Gene", "QC Metric"])

    with col2:
        if color_type == "Cluster":
            cluster_options = [c for c in obs_categorical if c in ['leiden', 'louvain', 'cluster']]
            if not cluster_options: cluster_options = obs_categorical
            color_col = st.selectbox("Select cluster column:", cluster_options if cluster_options else ['leiden'])
        elif color_type == "Marker Gene":
            cluster_col = _get_cluster_column(df)
            if not cluster_col:
                selected_gene = st.selectbox("Select gene:", sorted([c.replace('expr_', '') for c in expr_cols]))
                color_col = f"expr_{selected_gene}"
            else:
                mean_expr_by_cluster = df.groupby(cluster_col)[expr_cols].mean()
                gene_to_cluster = {col.replace('expr_', ''): mean_expr_by_cluster[col].idxmax() for col in expr_cols}
                gene_options_annotated = [f"{gene} (Cluster {gene_to_cluster[gene]})" for gene in sorted(gene_to_cluster.keys())]
                annotated_to_gene = {f"{gene} (Cluster {gene_to_cluster[gene]})": gene for gene in gene_to_cluster.keys()}
                selected_gene_annotated = st.selectbox("Select gene:", gene_options_annotated)
                selected_gene = annotated_to_gene[selected_gene_annotated]
                color_col = f"expr_{selected_gene}"
        else:
            metric_options = [c for c in obs_numerical if c in ['n_genes', 'n_counts', 'pct_counts_mt', 'n_genes_by_counts']]
            if not metric_options: metric_options = obs_numerical[:5] if obs_numerical else ['n_genes']
            color_col = st.selectbox("Select metric:", metric_options)

    with col3:
        point_size = st.slider("Point size:", 1, 10, 3)

    with st.expander("Advanced Options"):
        col_a, col_b = st.columns(2)
        with col_a: opacity = st.slider("Opacity:", 0.1, 1.0, 0.8)
        with col_b: color_scale = st.selectbox("Color scale:", ["Viridis", "Plasma", "Blues", "Reds", "RdBu", "Portland", "Turbo"])

    if 'UMAP_0' not in df.columns or 'UMAP_1' not in df.columns:
        st.warning("UMAP coordinates not found in data."); return

    is_categorical = color_col in obs_categorical or color_type == "Cluster"
    hover_data_dict = {'UMAP_0': ':.2f', 'UMAP_1': ':.2f'}
    for gene_col in expr_cols[:3]: hover_data_dict[gene_col] = ':.2f'

    if is_categorical:
        fig = px.scatter(df, x='UMAP_0', y='UMAP_1', color=color_col, hover_data=hover_data_dict,
                         title=f"UMAP colored by {color_col}", width=900, height=650)
    else:
        fig = px.scatter(df, x='UMAP_0', y='UMAP_1', color=color_col, color_continuous_scale=color_scale.lower(),
                         hover_data=hover_data_dict, title=f"UMAP colored by {color_col}", width=900, height=650)

    fig.update_traces(marker=dict(size=point_size, opacity=opacity, line=dict(width=0)))
    fig.update_layout(plot_bgcolor='white',
                      xaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 1'),
                      yaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 2'), font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    # Dataset Summary
    st.markdown("---")
    st.markdown("##### Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        try:
            client = MlflowClient()
            run_info = client.get_run(run_id)
            total_cells_actual = run_info.data.metrics.get('total_cells_before_subsample', None)
            if total_cells_actual:
                st.metric("Total Cells (Full)", f"{int(total_cells_actual):,}")
                st.caption(f"Viewing subsample of: {len(df):,}")
            else:
                st.metric("Total Cells", f"{len(df):,}*"); st.caption("*Subsampled")
        except:
            st.metric("Total Cells", f"{len(df):,}*"); st.caption("*Subsampled")
    with col2:
        cluster_col = _get_cluster_column(df)
        if cluster_col: st.metric("Clusters", df[cluster_col].nunique())
    with col3: st.metric("Marker Genes", len(expr_cols))
    with col4:
        if 'UMAP_0' in df.columns: st.metric("Embeddings", "UMAP")

    # Dot plot
    st.markdown("---")
    with st.expander("Marker Gene Expression by Cluster", expanded=False):
        cluster_col = _get_cluster_column(df)
        if cluster_col and expr_cols:
            try: marker_mapping = download_cluster_markers_mapping(run_id)
            except: marker_mapping = None

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                n_top_genes_dot = st.number_input("Top genes per cluster:", min_value=1, max_value=20, value=3)
                ordered_genes_by_cluster = []
                if marker_mapping is not None:
                    for cluster_id in sorted(marker_mapping.columns, key=lambda x: int(x) if x.isdigit() else x):
                        ordered_genes_by_cluster.extend(marker_mapping[cluster_id].head(n_top_genes_dot).dropna().tolist())
                else:
                    mean_expr = df.groupby(cluster_col)[expr_cols].mean()
                    mean_z = (mean_expr - mean_expr.mean()) / mean_expr.std()
                    for cluster in sorted(mean_z.index):
                        ordered_genes_by_cluster.extend([g.replace('expr_', '') for g in mean_z.loc[cluster].nlargest(n_top_genes_dot).index])
                ordered_genes = list(dict.fromkeys(ordered_genes_by_cluster))
                selected_genes = st.multiselect("Customize gene selection:", [c.replace('expr_', '') for c in expr_cols], default=ordered_genes)
            with col2: st.write(""); st.write(""); scale_data = st.checkbox("Scale expression", value=True)
            with col3: st.write(""); st.write(""); font_size = st.slider("Font size:", 10, 20, 14)

            if selected_genes:
                expr_cols_to_plot = [f"expr_{g}" for g in selected_genes]
                genes_ordered = selected_genes
            else:
                expr_cols_to_plot = expr_cols
                genes_ordered = [c.replace('expr_', '') for c in expr_cols]

            heatmap_data = df.groupby(cluster_col)[expr_cols_to_plot].mean()
            heatmap_data.columns = [c.replace('expr_', '') for c in heatmap_data.columns]
            heatmap_data = heatmap_data[genes_ordered]
            if scale_data:
                heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
                color_label, cs = "Z-score", "RdBu_r"
            else:
                color_label, cs = "Mean Expression", "Viridis"

            dotplot_data = [{'Cluster': str(cl), 'Gene': gene, 'Expression': heatmap_data.loc[cl, gene]}
                           for cl in heatmap_data.index for gene in heatmap_data.columns]
            dotplot_df = pd.DataFrame(dotplot_data)
            dotplot_df['Size'] = dotplot_df['Expression'].abs() if scale_data else dotplot_df['Expression']

            fig_dotplot = px.scatter(dotplot_df, x='Gene', y='Cluster', color='Expression', size='Size',
                                    color_continuous_scale=cs, labels={'Expression': color_label},
                                    title=f"Marker Expression by Cluster ({color_label})", height=max(400, len(heatmap_data.index) * 50))
            fig_dotplot.update_traces(marker=dict(sizemode='diameter', sizeref=dotplot_df['Size'].max() / 15, line=dict(width=0.5, color='white')))
            fig_dotplot.update_xaxes(tickangle=45, tickfont=dict(size=font_size))
            fig_dotplot.update_yaxes(tickfont=dict(size=font_size))
            fig_dotplot.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'),
                                     yaxis=dict(showgrid=True, gridcolor='lightgray'), font=dict(size=font_size))
            st.plotly_chart(fig_dotplot, use_container_width=True)
        else:
            st.warning("Clustering information or marker genes not available")

    # QC section
    st.markdown("---")
    with st.expander("QC & Other Analysis Outputs", expanded=False):
        st.info("All analysis outputs (QC plots, PCA, highly variable genes, marker genes heatmap, UMAP, etc.) are available in the MLflow run.")
        st.link_button("Open MLflow Run (View All Plots & Artifacts)", mlflow_url, type="primary")
        st.markdown("---")
        st.markdown("**About the MLflow Run:**\n"
                    "- Quality control plots\n- PCA and variance explained plots\n- Highly variable genes\n"
                    "- Full-resolution UMAP\n- Marker genes heatmap\n- Complete AnnData object")

    # Raw data
    st.markdown("---")
    with st.expander("View Raw Data Table"):
        cluster_col = _get_cluster_column(df)
        default_cols = [cluster_col, 'UMAP_0', 'UMAP_1'] if cluster_col else ['UMAP_0', 'UMAP_1']
        default_cols += expr_cols[:3]
        display_cols = st.multiselect("Select columns:", df.columns.tolist(), default=[c for c in default_cols if c in df.columns])
        if display_cols:
            st.dataframe(df[display_cols].head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df):,} cells")

    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv_data, file_name=f"singlecell_results_{st.session_state['singlecell_run_id'][:8]}.csv", mime="text/csv")
    with col3:
        st.link_button("MLflow Run", mlflow_url, type="secondary")


def render():
    st.markdown("### Raw Single Cell Analysis")

    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

    run_tab, view_tab = st.tabs(["Run New Analysis", "View Analysis Results"])

    with run_tab:
        _display_run_analysis()

    with view_tab:
        _display_results_viewer()
