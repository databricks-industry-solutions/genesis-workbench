
import streamlit as st
import pandas as pd
import time
import os
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models)

from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog,
                                    get_user_info)
from utils.single_cell_analysis import (start_scanpy_job, 
                                        start_rapids_singlecell_job,
                                        download_singlecell_markers_df,
                                        download_cluster_markers_mapping,
                                        get_mlflow_run_url,
                                        search_singlecell_runs)
import plotly.express as px
from mlflow.tracking import MlflowClient

def reset_available_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["available_single_cell_models_df"]
        st.rerun()

def reset_deployed_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["deployed_single_cell_models_df"]
        st.rerun()

def display_settings_tab(available_models_df,deployed_models_df):

    p1,p2 = st.columns([2,1])

    with p1:
        st.markdown("###### Import Models:")
        with st.form("import_model_form"):
            col1, col2, = st.columns([1,1], vertical_alignment="bottom")    
            with col1:
                import_model_source = st.selectbox("Source:",["Unity Catalog","Hugging Face","PyPi"],label_visibility="visible")

            with col2:
                import_button = st.form_submit_button('Import')
        
        if import_button:
            if import_model_source=="Unity Catalog":
                display_import_model_uc_dialog(ModelCategory.SINGLE_CELL, success_callback=reset_available_models)


        st.markdown("###### Available Models:")
        with st.form("deploy_model_form"):
            col1, col2, = st.columns([1,1])    
            with col1:
                selected_model_for_deploy = st.selectbox("Model:",available_models_df["model_labels"],label_visibility="collapsed",)

            with col2:
                deploy_button = st.form_submit_button('Deploy')
        if deploy_button:
            display_deploy_model_dialog(selected_model_for_deploy)


    if len(deployed_models_df) > 0:
        with st.form("modify_deployed_model_form"):
            col1,col2 = st.columns([2,1])
            with col1:
                st.markdown("###### Deployed Models")
            with col2:
                st.form_submit_button("Manage")
            
            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
    else:
        st.write("There are no deployed models")


def display_scanpy_analysis_tab():
    st.markdown("###### Run Scanpy Analysis")
    
    with st.form("scanpy_analysis_form", enter_to_submit=False):
        # Mode Selection
        st.markdown("**Analysis Mode:**")
        mode_display = st.selectbox(
            "Mode",
            options=["scanpy", "rapids-singlecell [GPU-accelerated]"],
            label_visibility="collapsed"
        )
        # Extract actual mode name
        mode = "rapids-singlecell" if "rapids-singlecell" in mode_display else mode_display
        
        st.divider()
        
        # Data Input
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Data Configuration:**")
            data_path = st.text_input(
                "Data Path (h5ad file)",
                placeholder="/Volumes/catalog/schema/volume/file.h5ad",
                help="Path to the h5ad file in Unity Catalog Volumes"
            )
            gene_name_column = st.text_input(
                "Gene Name Column (optional)",
                value="",
                placeholder="e.g., gene_name, feature_name",
                help="Name of column in var containing gene names. Leave empty to use Ensembl reference mapping. Note: Gene names will be normalized to uppercase for consistent QC analysis."
            )
            
            # Conditional species selector
            if not gene_name_column or gene_name_column.strip() == "":
                species = st.selectbox(
                    "Species",
                    options=["hsapiens", "mmusculus", "rnorvegicus"],
                    index=0,
                    help="Species for Ensembl gene name mapping. Required when gene name column is not provided. Note: All gene names are normalized to uppercase for consistent QC detection (mitochondrial, ribosomal genes) regardless of input capitalization.",
                    format_func=lambda x: {
                        "hsapiens": "Human (Homo sapiens)",
                        "mmusculus": "Mouse (Mus musculus)",
                        "rnorvegicus": "Rat (Rattus norvegicus)"
                    }[x]
                )
            else:
                species = "hsapiens"  # Default, won't be used since gene_name_column is provided
            
            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input(
                "MLflow Experiment Name",
                value="scanpy_genesis_workbench",
                help="Simple experiment name (will be created in your MLflow folder)"
            )
            mlflow_run_name = st.text_input(
                "MLflow Run Name",
                value="scanpy_analysis",
                help="Name for this specific analysis run"
            )
        
        with col2:
            st.markdown("**Filtering Parameters:**")
            min_genes = st.number_input(
                "Min Genes per Cell",
                min_value=0,
                value=200,
                step=10
            )
            min_cells = st.number_input(
                "Min Cells per Gene",
                min_value=0,
                value=3,
                step=1
            )
            pct_counts_mt = st.number_input(
                "Max % Mitochondrial Counts",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1
            )
            n_genes_by_counts = st.number_input(
                "Max Genes by Counts",
                min_value=0,
                value=2500,
                step=100
            )
        
        st.divider()
        
        # Analysis Parameters
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("**Normalization & Feature Selection:**")
            target_sum = st.number_input(
                "Target Sum for Normalization",
                min_value=0,
                value=10000,
                step=1000
            )
            n_top_genes = st.number_input(
                "Number of Highly Variable Genes",
                min_value=0,
                value=500,
                step=50
            )
        
        with col4:
            st.markdown("**Dimensionality Reduction & Clustering:**")
            n_pcs = st.number_input(
                "Number of Principal Components",
                min_value=0,
                value=50,
                step=5
            )
            cluster_resolution = st.number_input(
                "Cluster Resolution",
                min_value=0.0,
                max_value=2.0,
                value=0.15,
                step=0.05,
                format="%.2f"
            )
        
        st.divider()
        submit_button = st.form_submit_button("Start Analysis", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        is_valid = True
        
        # Validation
        if not data_path.strip():
            st.error("Please provide a data path")
            is_valid = False
        elif not data_path.endswith(".h5ad"):
            st.error("Data path must point to an .h5ad file")
            is_valid = False
        elif not data_path.startswith("/Volumes"):
            st.error("Data path must start with /Volumes (Unity Catalog Volume)")
            is_valid = False
        
        if not mlflow_experiment.strip() or not mlflow_run_name.strip():
            st.error("Please provide MLflow experiment and run names")
            is_valid = False
        
        # Validate gene name column or species is provided
        if (not gene_name_column or gene_name_column.strip() == "") and (not species or species == ""):
            st.error("❌ Please either provide a Gene Name Column OR select a Species for reference mapping.")
            is_valid = False
        
        if is_valid:
            user_info = get_user_info()
            
            if mode == "scanpy":
                try:
                    with st.spinner("Starting scanpy analysis job..."):
                        scanpy_job_id, job_run_id = start_scanpy_job(
                            data_path=data_path,
                            mlflow_experiment=mlflow_experiment,
                            mlflow_run_name=mlflow_run_name,
                            gene_name_column=gene_name_column,
                            species=species,
                            min_genes=min_genes,
                            min_cells=min_cells,
                            pct_counts_mt=pct_counts_mt,
                            n_genes_by_counts=n_genes_by_counts,
                            target_sum=target_sum,
                            n_top_genes=n_top_genes,
                            n_pcs=n_pcs,
                            cluster_resolution=cluster_resolution,
                            user_info=user_info
                        )
                        
                        # Construct the run URL
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/{scanpy_job_id}/runs/{job_run_id}"
                        
                        st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                        st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                
                except Exception as e:
                    st.error(f"❌ An error occurred while starting the job: {str(e)}")
                    print(e)
            
            elif mode == "rapids-singlecell":
                try:
                    with st.spinner("Starting rapids-singlecell analysis job..."):
                        rapids_job_id, job_run_id = start_rapids_singlecell_job(
                            data_path=data_path,
                            mlflow_experiment=mlflow_experiment,
                            mlflow_run_name=mlflow_run_name,
                            gene_name_column=gene_name_column,
                            species=species,
                            min_genes=min_genes,
                            min_cells=min_cells,
                            pct_counts_mt=pct_counts_mt,
                            n_genes_by_counts=n_genes_by_counts,
                            target_sum=target_sum,
                            n_top_genes=n_top_genes,
                            n_pcs=n_pcs,
                            cluster_resolution=cluster_resolution,
                            user_info=user_info
                        )
                        
                        # Construct the run URL
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/{rapids_job_id}/runs/{job_run_id}"
                        
                        st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                        st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                
                except Exception as e:
                    st.error(f"❌ An error occurred while starting the job: {str(e)}")
                    print(e)
            
            else:
                st.error(f"❌ Unknown mode: {mode}")


def display_singlecell_results_viewer():
    """Interactive viewer for single-cell analysis results (scanpy, rapids-singlecell, etc.)"""
    
    # Helper function to find cluster column (supports legacy and new naming)
    def get_cluster_column(df):
        """Return the cluster column name (prioritizes 'cluster', falls back to 'leiden' or 'louvain')"""
        if 'cluster' in df.columns:
            return 'cluster'
        elif 'leiden' in df.columns:
            return 'leiden'
        elif 'louvain' in df.columns:
            return 'louvain'
        return None
    
    st.markdown("##### Single-Cell Results Viewer")
    st.markdown("Select a completed single-cell analysis run to visualize")
    
    # Get user info
    user_info = get_user_info()
    
    # Initialize session state for filters
    if 'date_filter' not in st.session_state:
        st.session_state['date_filter'] = None  # All time by default
    if 'processing_mode_filter' not in st.session_state:
        st.session_state['processing_mode_filter'] = "All"
    
    # Compact top row: Experiment (50%), Mode (15%), Time filters (25%), Refresh (10%)
    main_col1, main_col2, main_col3, main_col4 = st.columns([5, 1.5, 2.5, 1], vertical_alignment="bottom")
    
    with main_col1:
        experiment_filter = st.text_input(
            "MLflow Experiment:",
            value="scanpy_genesis_workbench",
            help="Enter experiment name to filter runs (partial match supported)."
        )
    
    with main_col2:
        processing_mode_option = st.radio(
            "Mode:",
            ["All", "Scanpy", "Rapids-SingleCell"],
            index=["All", "Scanpy", "Rapids-SingleCell"].index(st.session_state['processing_mode_filter']),
            help="Filter by processing pipeline",
            label_visibility="visible"
        )
        if processing_mode_option != st.session_state['processing_mode_filter']:
            st.session_state['processing_mode_filter'] = processing_mode_option
            if 'singlecell_runs_df' in st.session_state:
                del st.session_state['singlecell_runs_df']
    
    with main_col3:
        st.markdown("**Time Period:**")
        # 2x2 grid for time filters
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if st.button("Today", use_container_width=True, help="Show runs from today"):
                st.session_state['date_filter'] = 0
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        with row1_col2:
            if st.button("Last 7 Days", use_container_width=True, help="Show runs from last week"):
                st.session_state['date_filter'] = 7
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if st.button("Last 30 Days", use_container_width=True, help="Show runs from last month"):
                st.session_state['date_filter'] = 30
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        with row2_col2:
            if st.button("All Time", use_container_width=True, help="Show all runs"):
                st.session_state['date_filter'] = None
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
    
    with main_col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        refresh_button = st.button("🔄 Refresh", use_container_width=True, help="Reload available runs from MLflow")
    
    # Load or refresh runs list
    if refresh_button or 'singlecell_runs_df' not in st.session_state:
        with st.spinner("Searching for your single-cell analysis runs..."):
            try:
                # Convert processing mode to lowercase for tag matching
                processing_mode = None
                if st.session_state['processing_mode_filter'] != "All":
                    processing_mode = st.session_state['processing_mode_filter'].lower().replace("-", "-")
                    if processing_mode == "rapids-singlecell":
                        processing_mode = "rapids-singlecell"
                    elif processing_mode == "scanpy":
                        processing_mode = "scanpy"
                
                runs_df = search_singlecell_runs(
                    user_email=user_info.user_email,
                    processing_mode=processing_mode,
                    days_back=st.session_state['date_filter']
                )
                st.session_state['singlecell_runs_df'] = runs_df
                if len(runs_df) == 0:
                    st.info("No single-cell analysis runs found. Run an analysis first!")
            except Exception as e:
                st.error(f"❌ Error searching runs: {str(e)}")
                return
    
    runs_df = st.session_state.get('singlecell_runs_df', pd.DataFrame())
    
    if len(runs_df) == 0:
        st.info("💡 No single-cell runs found. Go to 'Run New Analysis' to create one!")
        return
    
    # Apply experiment text filter
    if experiment_filter and experiment_filter.strip():
        runs_df = runs_df[runs_df['experiment'].str.contains(experiment_filter.strip(), case=False, na=False)]
        if len(runs_df) == 0:
            st.warning(f"No runs found matching experiment: '{experiment_filter}'")
            return
    
    st.divider()
    
    # Sort by most recent (default)
    runs_df = runs_df.sort_values('start_time', ascending=False)
    
    # Create display names with date and status
    runs_df['display_name'] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1
    )
    
    run_options = dict(zip(runs_df['display_name'], runs_df['run_id']))
    
    # Consolidated run selection: Run name filter (40%), Runs dropdown (50%), Load button (10%)
    select_col1, select_col2, select_col3 = st.columns([4, 5, 1], vertical_alignment="bottom")
    
    with select_col1:
        run_name_filter = st.text_input(
            "Search by Run Name:",
            value="",
            placeholder="Type to filter runs...",
            help="Enter run name to filter runs (partial match supported)."
        )
    
    # Apply run name text filter
    if run_name_filter and run_name_filter.strip():
        filtered_runs = runs_df[runs_df['run_name'].str.contains(run_name_filter.strip(), case=False, na=False)]
        if len(filtered_runs) == 0:
            st.warning(f"No runs found matching run name: '{run_name_filter}'")
            return
        filtered_options = dict(zip(filtered_runs['display_name'], filtered_runs['run_id']))
    else:
        filtered_runs = runs_df
        filtered_options = run_options
    
    with select_col2:
        st.markdown(f"**{len(filtered_runs)} Runs:**")
        selected_display = st.selectbox(
            "Select:",
            list(filtered_options.keys()),
            help="Runs sorted by most recent. Experiment name in parentheses.",
            label_visibility="collapsed"
        )
    
    with select_col3:
        st.write("")  # Spacer to align button
        load_button = st.button("Load", type="primary", use_container_width=True)
    
    run_id = filtered_options[selected_display]
    
    # Load button action
    if load_button:
        with st.spinner("Loading data from MLflow..."):
            try:
                df = download_singlecell_markers_df(run_id)
                mlflow_url = get_mlflow_run_url(run_id)
                
                st.session_state['singlecell_df'] = df
                st.session_state['singlecell_run_id'] = run_id
                st.session_state['singlecell_mlflow_url'] = mlflow_url
                
                st.success(f"✅ Loaded {len(df):,} cells with {len(df.columns)} features")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("💡 Tip: Make sure this run includes the markers_flat.parquet artifact")
                return
    
    # Step 3: Display if data is loaded
    if 'singlecell_df' in st.session_state:
        df = st.session_state['singlecell_df']
        run_id = st.session_state.get('singlecell_run_id', '')
        mlflow_url = st.session_state.get('singlecell_mlflow_url', '')
        
        st.markdown("---")
        st.markdown("##### Interactive UMAP Visualization")
        
        # Info about subsampled data
        st.info(
            "📊 **Note:** This viewer displays a subsampled dataset (max 10,000 cells) with marker genes only "
            "for faster, interactive plotting. The complete output AnnData object with all genes is available "
            "in the MLflow run (see 'QC & Other Analysis Outputs' section below)."
        )
        
        # Identify column types
        expr_cols = [c for c in df.columns if c.startswith('expr_')]
        obs_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        obs_numerical = df.select_dtypes(include=['number']).columns.tolist()
        # Remove UMAP/PC columns from numerical
        obs_numerical = [c for c in obs_numerical if not c.startswith(('UMAP_', 'PC_'))]
        
        # Controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            color_type = st.selectbox(
                "Color by:",
                ["Cluster", "Marker Gene", "QC Metric"],
                help="Choose what to color the cells by"
            )
        
        with col2:
            if color_type == "Cluster":
                cluster_options = [c for c in obs_categorical if c in ['leiden', 'louvain', 'cluster']]
                if not cluster_options:
                    cluster_options = obs_categorical
                color_col = st.selectbox("Select cluster column:", cluster_options if cluster_options else ['leiden'])
            elif color_type == "Marker Gene":
                # Calculate which cluster each gene is most expressed in
                cluster_col = get_cluster_column(df)
                if not cluster_col:
                    st.warning("⚠️ No cluster column found in data. Cannot annotate genes by cluster.")
                    # Fallback: just use gene names without cluster annotation
                    selected_gene = st.selectbox("Select gene:", sorted([c.replace('expr_', '') for c in expr_cols]))
                    color_col = f"expr_{selected_gene}"
                else:
                    mean_expr_by_cluster = df.groupby(cluster_col)[expr_cols].mean()
                    
                    # For each gene, find the cluster with highest mean expression
                    gene_to_cluster = {}
                    for gene_col in expr_cols:
                        gene_name = gene_col.replace('expr_', '')
                        cluster_with_max = mean_expr_by_cluster[gene_col].idxmax()
                        gene_to_cluster[gene_name] = cluster_with_max
                    
                    # Create annotated gene options: "GENE (Cluster X)"
                    gene_options_annotated = [
                        f"{gene} (Cluster {gene_to_cluster[gene]})" 
                        for gene in sorted(gene_to_cluster.keys())
                    ]
                    
                    # Create a mapping for reverse lookup
                    annotated_to_gene = {
                        f"{gene} (Cluster {gene_to_cluster[gene]})": gene
                        for gene in gene_to_cluster.keys()
                    }
                    
                    selected_gene_annotated = st.selectbox("Select gene:", gene_options_annotated)
                    # Extract the actual gene name from the annotated selection
                    selected_gene = annotated_to_gene[selected_gene_annotated]
                    color_col = f"expr_{selected_gene}"
            else:  # QC Metric
                metric_options = [c for c in obs_numerical if c in ['n_genes', 'n_counts', 'pct_counts_mt', 'n_genes_by_counts']]
                if not metric_options:
                    metric_options = obs_numerical[:5] if len(obs_numerical) > 0 else ['n_genes']
                color_col = st.selectbox("Select metric:", metric_options)
        
        with col3:
            point_size = st.slider("Point size:", 1, 10, 3)
        
        # Additional options
        with st.expander("⚙️ Advanced Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.8)
            with col_b:
                color_scale = st.selectbox(
                    "Color scale:",
                    ["Viridis", "Plasma", "Blues", "Reds", "RdBu", "Portland", "Turbo"]
                )
        
        # Create the plot
        if 'UMAP_0' not in df.columns or 'UMAP_1' not in df.columns:
            st.warning("⚠️ UMAP coordinates not found in data. Cannot display plot.")
        else:
            # Determine if categorical or continuous
            is_categorical = color_col in obs_categorical or color_type == "Cluster"
            
            # Prepare hover data
            hover_data_dict = {
                'UMAP_0': ':.2f',
                'UMAP_1': ':.2f',
            }
            # Add a few marker genes to hover
            for gene_col in expr_cols[:3]:
                hover_data_dict[gene_col] = ':.2f'
            
            if is_categorical:
                fig = px.scatter(
                    df,
                    x='UMAP_0',
                    y='UMAP_1',
                    color=color_col,
                    hover_data=hover_data_dict,
                    title=f"UMAP colored by {color_col}",
                    width=900,
                    height=650
                )
            else:
                fig = px.scatter(
                    df,
                    x='UMAP_0',
                    y='UMAP_1',
                    color=color_col,
                    color_continuous_scale=color_scale.lower(),
                    hover_data=hover_data_dict,
                    title=f"UMAP colored by {color_col}",
                    width=900,
                    height=650
                )
            
            # Update layout
            fig.update_traces(
                marker=dict(size=point_size, opacity=opacity, line=dict(width=0))
            )
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 1'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 2'),
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats - moved below plot
        st.markdown("---")
        st.markdown("##### 📊 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Get total cells from MLflow metrics if available
            try:
                client = MlflowClient()
                run_info = client.get_run(run_id)
                total_cells_actual = run_info.data.metrics.get('total_cells_before_subsample', None)
                if total_cells_actual:
                    st.metric("Total Cells (Full)", f"{int(total_cells_actual):,}")
                    st.caption(f"Viewing subsample of: {len(df):,}")
                else:
                    st.metric("Total Cells", f"{len(df):,}*")
                    st.caption("*Subsampled")
            except:
                st.metric("Total Cells", f"{len(df):,}*")
                st.caption("*Subsampled")
        
        with col2:
            cluster_col = get_cluster_column(df)
            if cluster_col:
                st.metric("Clusters", df[cluster_col].nunique())
        with col3:
            st.metric("Marker Genes", len(expr_cols))
        with col4:
            if 'UMAP_0' in df.columns:
                st.metric("Embeddings", "UMAP ✓")
        
        # Dot plot section
        st.markdown("---")
        with st.expander("Marker Gene Expression by Cluster", expanded=False):
            cluster_col = get_cluster_column(df)
            if cluster_col and expr_cols:
                # Load the cluster-to-marker mapping from MLflow
                try:
                    marker_mapping = download_cluster_markers_mapping(run_id)
                except:
                    marker_mapping = None
                
                # Gene selection - calculate top genes per cluster
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    n_top_genes = st.number_input(
                        "Top genes per cluster:",
                        min_value=1,
                        max_value=20,
                        value=3,
                        help="Select top N marker genes per cluster (ranked by Wilcoxon test)"
                    )
                    
                    # Get ordered list of genes using Wilcoxon rankings
                    ordered_genes_by_cluster = []
                    
                    if marker_mapping is not None:
                        # Use the pre-computed marker rankings from scanpy
                        for cluster_id in sorted(marker_mapping.columns, key=lambda x: int(x) if x.isdigit() else x):
                            # Get top N genes for this cluster (already ranked by Wilcoxon)
                            top_genes = marker_mapping[cluster_id].head(n_top_genes).dropna().tolist()
                            ordered_genes_by_cluster.extend(top_genes)
                    else:
                        # Fallback: use z-score ranking if CSV not available
                        mean_expr_by_cluster = df.groupby(cluster_col)[expr_cols].mean()
                        mean_expr_zscored = (mean_expr_by_cluster - mean_expr_by_cluster.mean()) / mean_expr_by_cluster.std()
                        
                        for cluster in sorted(mean_expr_zscored.index):
                            cluster_zscores = mean_expr_zscored.loc[cluster]
                            top_genes = cluster_zscores.nlargest(n_top_genes).index.tolist()
                            top_genes = [g.replace('expr_', '') for g in top_genes]
                            ordered_genes_by_cluster.extend(top_genes)
                    
                    # Remove duplicates while preserving order
                    ordered_genes = []
                    seen = set()
                    for gene in ordered_genes_by_cluster:
                        if gene not in seen:
                            ordered_genes.append(gene)
                            seen.add(gene)
                    
                    # Allow user to customize selection
                    selected_genes = st.multiselect(
                        "Customize gene selection:",
                        [c.replace('expr_', '') for c in expr_cols],
                        default=ordered_genes,
                        help="Pre-populated with top genes per cluster (ordered by cluster)"
                    )
                
                with col2:
                    st.write("")
                    st.write("")
                    scale_data = st.checkbox("Scale expression", value=True, help="Z-score normalization (enhances relative differences)")
                
                with col3:
                    st.write("")
                    st.write("")
                    font_size = st.slider("Font size:", 10, 20, 14)
                
                if selected_genes:
                    # Maintain the cluster-ordered gene list
                    expr_cols_to_plot = [f"expr_{g}" for g in selected_genes]
                    genes_ordered = selected_genes  # Keep for x-axis ordering
                else:
                    expr_cols_to_plot = expr_cols
                    genes_ordered = [c.replace('expr_', '') for c in expr_cols]
                
                # Calculate mean expression per cluster
                heatmap_data = df.groupby(cluster_col)[expr_cols_to_plot].mean()
                heatmap_data.columns = [c.replace('expr_', '') for c in heatmap_data.columns]
                
                # Reorder columns to match cluster-based ordering
                heatmap_data = heatmap_data[genes_ordered]
                
                # Optional scaling
                if scale_data:
                    heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
                    color_label = "Z-score"
                    color_scale = "RdBu_r"
                else:
                    color_label = "Mean Expression"
                    color_scale = "Viridis"
                
                # Prepare data for dot plot (convert from wide to long format)
                dotplot_data = []
                for cluster in heatmap_data.index:
                    for gene in heatmap_data.columns:
                        dotplot_data.append({
                            'Cluster': str(cluster),
                            'Gene': gene,
                            'Expression': heatmap_data.loc[cluster, gene]
                        })
                
                dotplot_df = pd.DataFrame(dotplot_data)
                
                # Create dot plot
                # For size, use absolute values or original expression (size must be non-negative)
                if scale_data:
                    # When z-scored, use absolute values for size, but keep z-scores for color
                    dotplot_df['Size'] = dotplot_df['Expression'].abs()
                else:
                    dotplot_df['Size'] = dotplot_df['Expression']
                
                fig_dotplot = px.scatter(
                    dotplot_df,
                    x='Gene',
                    y='Cluster',
                    color='Expression',  # Color by z-score (can be negative)
                    size='Size',  # Size by absolute value (always positive)
                    color_continuous_scale=color_scale,
                    labels={'Expression': color_label},
                    title=f"Marker Expression by Cluster ({color_label})",
                    height=max(400, len(heatmap_data.index) * 50)
                )
                
                # Update layout for better readability
                fig_dotplot.update_traces(
                    marker=dict(
                        sizemode='diameter',
                        sizeref=dotplot_df['Size'].max() / 15,  # Adjust dot size scaling
                        line=dict(width=0.5, color='white')
                    )
                )
                fig_dotplot.update_xaxes(tickangle=45, tickfont=dict(size=font_size))
                fig_dotplot.update_yaxes(tickfont=dict(size=font_size))
                fig_dotplot.update_layout(
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    font=dict(size=font_size),
                    title=dict(font=dict(size=font_size + 2))
                )
                
                st.plotly_chart(fig_dotplot, use_container_width=True)
            else:
                st.warning("Clustering information or marker genes not available")
        
        # QC and additional analysis section
        st.markdown("---")
        with st.expander("QC & Other Analysis Outputs", expanded=False):
            st.info(
                "**View Standard Analysis Plots**: All analysis outputs (QC plots, "
                "PCA, highly variable genes, marker genes heatmap, UMAP, etc.) are available in the MLflow run."
                "We may add QC interactive vizualization here in the future."
            )
            st.link_button("🔗 Open MLflow Run (View All Plots & Artifacts)", mlflow_url, type="primary")
            
            st.markdown("---")
            st.markdown("**About the MLflow Run:**")
            st.markdown(
                "- Quality control plots (cell/gene filtering, mitochondrial content)\n"
                "- PCA and variance explained plots\n"
                "- Highly variable genes identification\n"
                "- Full-resolution UMAP (all cells, all genes)\n"
                "- Marker genes heatmap\n"
                "- Complete AnnData object with all genes"
            )
        
        # Data table preview
        st.markdown("---")
        with st.expander("📊 View Raw Data Table"):
            # Show selected columns
            cluster_col = get_cluster_column(df)
            default_cols = [cluster_col, 'UMAP_0', 'UMAP_1'] if cluster_col else ['UMAP_0', 'UMAP_1']
            default_cols += expr_cols[:3]
            
            display_cols = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=[c for c in default_cols if c in df.columns]
            )
            if display_cols:
                st.dataframe(df[display_cols].head(100), use_container_width=True)
                st.caption(f"Showing first 100 of {len(df):,} cells")
        
        # Download section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"singlecell_results_{st.session_state['singlecell_run_id'][:8]}.csv",
                mime="text/csv"
            )
        with col3:
            st.link_button("🔗 MLflow Run", mlflow_url, type="secondary")


def display_embeddings_tab(deployed_models_df):
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("###### Generate Embeddings")
    with col2:        
        st.button("View Past Runs")    

    if len(deployed_models_df) > 0:
        with st.form("run_embedding_form"):
            st.write("Select Models:")


            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="multi-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
        
            st.write("NOTE: A result table will be created for EACH model selected.")

            col1, col2, col3 = st.columns([1,1,1], vertical_alignment="bottom")
            with col1:        
                st.text_input("Data Location:","")
                st.text_input("Result Schema Name:","")
                st.text_input("Result Table Prefix:","")
            
            with col2:
                st.write("")
                st.toggle("Perform Evaluation?")            
                st.text_input("Ground Truth Data Location:","")
                st.text_input("MLflow Experiment Name:","")
            
            st.form_submit_button("Generate Embeddings")

    else:
        st.write("There are no deployed models")

#load data for page
with st.spinner("Loading data"):
    if "available_single_cell_models_df" not in st.session_state:
            available_single_cell_models_df = get_available_models(ModelCategory.SINGLE_CELL)
            available_single_cell_models_df["model_labels"] = (available_single_cell_models_df["model_id"].astype(str) + " - " 
                                                + available_single_cell_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_single_cell_models_df["model_uc_name"].astype(str) + " v"
                                                + available_single_cell_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_single_cell_models_df"] = available_single_cell_models_df
    available_single_cell_models_df = st.session_state["available_single_cell_models_df"]

    if "deployed_single_cell_models_df" not in st.session_state:
        deployed_single_cell_models_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        deployed_single_cell_models_df.columns = ["Model Id","Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]

        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]



st.title(":material/microbiology:  Single Cell Studies")

# settings_tab, processing_tab, embeddings_tab = st.tabs([
#     "Settings", 
#     "Raw Single Cell Processing",
#     "Embeddings"
# ])

settings_tab, processing_tab = st.tabs([
    "Settings", 
    "Raw Single Cell Processing",
    # "Embeddings"
])


with settings_tab:
    display_settings_tab(available_single_cell_models_df,deployed_single_cell_models_df)

with processing_tab:
    # Sub-sections within processing tab
    st.markdown("### Raw Single Cell Analysis")
    
    # Custom CSS to make tab text bigger
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use tabs instead of nested expanders to avoid nesting issues
    run_tab, view_tab = st.tabs(["Run New Analysis", "View Analysis Results"])
    
    with run_tab:
        display_scanpy_analysis_tab()
    
    with view_tab:
        display_singlecell_results_viewer()

# with embeddings_tab:
#     display_embeddings_tab(deployed_single_cell_models_df)
