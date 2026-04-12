"""Disease Biology — GWAS analysis with Parabricks + Glow, VCF ingestion, variant annotation."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import math
import matplotlib
import matplotlib.pyplot as plt

from utils.disease_biology import (
    start_parabricks_alignment,
    start_gwas_analysis,
    search_variant_calling_runs_by_run_name,
    search_variant_calling_runs_by_experiment_name,
    search_gwas_runs_by_run_name,
    search_gwas_runs_by_experiment_name,
    list_successful_variant_calling_runs,
    pull_gwas_results,
    start_vcf_ingestion,
    search_vcf_ingestion_runs_by_run_name,
    search_vcf_ingestion_runs_by_experiment_name,
    list_successful_vcf_ingestion_runs,
    start_variant_annotation,
    search_variant_annotation_runs_by_run_name,
    search_variant_annotation_runs_by_experiment_name,
    pull_annotation_results,
    add_progress_column,
    render_runs_html_table,
)
from utils.streamlit_helper import get_user_info, open_run_window


from datetime import datetime

st.title(":material/coronavirus: Disease Biology")

# Blinking dot CSS for in-progress runs
st.markdown("""
<style>
@keyframes blink-orange { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
.run-dot-in-progress { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                       background-color: #FF8C00; animation: blink-orange 1.2s infinite;
                       margin-right: 8px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)


def _show_in_progress_banner(df):
    """Show a blinking orange dot banner if any runs are in progress."""
    if df.empty or "status" not in df.columns:
        return
    in_progress = df[df["status"].isin(["started", "phenotype_prepared"])]
    if not in_progress.empty:
        count = len(in_progress)
        st.markdown(
            f'<span class="run-dot-in-progress"></span> '
            f'<strong>{count} run{"s" if count > 1 else ""} in progress</strong>',
            unsafe_allow_html=True,
        )


user_info = get_user_info()
_ts = datetime.now().strftime("%Y%m%d_%H%M")

alignment_tab, gwas_tab, ingestion_tab, annotation_tab = st.tabs([
    "Variant Calling",
    "GWAS Analysis",
    "VCF Ingestion",
    "Variant Annotation",
])


# ── Variant Calling result-row selection callback ──

def _set_selected_vc_row_status():
    selection = st.session_state["vc_run_search_result_display_df"].selection
    if len(selection["rows"]) > 0:
        idx = selection["rows"][0]
        status = st.session_state["vc_run_search_result_df"].iloc[idx]["status"]
        st.session_state["selected_vc_run_status"] = status
    else:
        st.session_state.pop("selected_vc_run_status", None)


@st.dialog("Variant Calling Results", width="large")
def _display_vc_results_dialog(selected_row_index):
    row = st.session_state["vc_run_search_result_df"].iloc[selected_row_index]
    run_id = row["run_id"]
    if isinstance(run_id, pd.Series):
        run_id = run_id.iloc[0]
    run_name = row["run_name"]
    if isinstance(run_name, pd.Series):
        run_name = run_name.iloc[0]

    st.markdown(f"##### Run: {run_name}")

    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    try:
        mlflow_run = mlflow.get_run(run_id)
        params = mlflow_run.data.params
        tags = mlflow_run.data.tags

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Status", tags.get("job_status", "unknown"))
        with mc2:
            st.metric("Job Run ID", tags.get("job_run_id", "N/A"))

        st.divider()

        st.markdown("**Input Files**")
        st.text(f"FASTQ R1:  {params.get('fastq_r1', 'N/A')}")
        st.text(f"FASTQ R2:  {params.get('fastq_r2', 'N/A')}")
        st.text(f"Reference: {params.get('reference_genome', 'N/A')}")

        st.divider()

        st.markdown("**Output Files**")
        output_bam = params.get("output_bam", "N/A")
        output_vcf = params.get("output_vcf", "N/A")
        st.text(f"BAM: {output_bam}")
        st.text(f"VCF: {output_vcf}")

        if output_vcf != "N/A":
            st.caption("Use this VCF path in the GWAS Analysis tab, or select this run directly via 'From Variant Calling Run'.")

    except Exception as e:
        st.error(f"Error loading run details: {e}")


# ── GWAS result-row selection callback ──

def _set_selected_gwas_row_status():
    selection = st.session_state["gwas_run_search_result_display_df"].selection
    if len(selection["rows"]) > 0:
        idx = selection["rows"][0]
        status = st.session_state["gwas_run_search_result_df"].iloc[idx]["status"]
        st.session_state["selected_gwas_run_status"] = status
    else:
        st.session_state.pop("selected_gwas_run_status", None)


@st.dialog("GWAS Results", width="large")
def _display_gwas_results_dialog(selected_row_index):
    row = st.session_state["gwas_run_search_result_df"].iloc[selected_row_index]
    run_id = row["run_id"]
    if isinstance(run_id, pd.Series):
        run_id = run_id.iloc[0]
    run_name = row["run_name"]
    if isinstance(run_name, pd.Series):
        run_name = run_name.iloc[0]

    st.markdown(f"##### Run: {run_name}")

    with st.spinner("Loading GWAS results..."):
        try:
            results_df = pull_gwas_results(run_id)

            if results_df.empty:
                st.warning("No significant results found for this run. "
                           "All p-values may be NULL — this can happen when the sample size is too small "
                           "or the phenotype has insufficient variation for the statistical test.")
                return

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Total Variants", f"{len(results_df):,}")
            with mc2:
                sig_count = len(results_df[results_df['pvalue'] < 5e-8])
                st.metric("Significant (p < 5e-8)", f"{sig_count:,}")
            with mc3:
                min_p = results_df['pvalue'].min()
                st.metric("Min p-value", f"{min_p:.2e}")

            st.divider()

            st.markdown("**Manhattan Plot**")
            matplotlib.use('Agg')

            fig, ax = plt.subplots(figsize=(12, 4))
            plot_df = results_df[results_df['neg_log_pval'].notna()].copy()
            if not plot_df.empty:
                ax.scatter(plot_df['start'], plot_df['neg_log_pval'],
                           s=2, alpha=0.5, c='#1f77b4')
                ax.axhline(y=-math.log10(5e-8), color='red',
                           linestyle='--', linewidth=0.8, label='Genome-wide (5e-8)')
                ax.axhline(y=5, color='blue',
                           linestyle='--', linewidth=0.5, label='Suggestive (1e-5)')
                ax.set_xlabel('Genomic Position')
                ax.set_ylabel('-log10(p-value)')
                ax.legend(fontsize=8)
                ax.set_title('GWAS Manhattan Plot')
            st.pyplot(fig)
            plt.close(fig)

            st.divider()

            st.markdown("**Top Hits**")
            top_hits = results_df.head(50)[['contigName', 'start', 'pvalue',
                                             'referenceAllele', 'alternateAlleles',
                                             'neg_log_pval']]
            top_hits.columns = ['Contig', 'Position', 'P-value',
                                'Ref Allele', 'Alt Alleles', '-log10(p)']
            st.dataframe(top_hits, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading results: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Variant Calling (Parabricks)
# ══════════════════════════════════════════════════════════════════════════════

with alignment_tab:
    st.markdown("### Variant Calling with Parabricks")
    st.markdown(
        "Run NVIDIA Parabricks "
        "`germline` pipeline on paired-end FASTQ files to produce aligned BAM and "
        "germline VCF output."
    )

    with st.form("alignment_form", enter_to_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            fastq_r1 = st.text_input(
                "FASTQ Read 1 (UC Volume path):",
                value=os.getenv("SAMPLE_FASTQ_R1", ""),
                placeholder="/Volumes/catalog/schema/volume/sample_R1.fq.gz",
                help="Path to paired-end read 1 FASTQ file in a UC Volume"
            )
        with c2:
            fastq_r2 = st.text_input(
                "FASTQ Read 2 (UC Volume path):",
                value=os.getenv("SAMPLE_FASTQ_R2", ""),
                placeholder="/Volumes/catalog/schema/volume/sample_R2.fq.gz",
                help="Path to paired-end read 2 FASTQ file in a UC Volume"
            )

        default_ref = os.getenv("GWAS_REFERENCE_GENOME_PATH", "")
        ref_choice = st.selectbox(
            "Reference Genome:",
            ["GRCh38 (pre-staged)", "Custom path"],
            help="GRCh38 is pre-staged during module setup. Select 'Custom path' to provide your own."
        )
        if ref_choice == "Custom path":
            reference_genome = st.text_input("Custom reference genome path:", "")
        else:
            reference_genome = default_ref
            st.caption(f"Using: `{default_ref}`")

        output_path = st.text_input(
            "Output Volume Path:",
            value=f"/Volumes/{os.getenv('CORE_CATALOG_NAME', 'catalog')}/{os.getenv('CORE_SCHEMA_NAME', 'schema')}/gwas_data",
            help="UC Volume path where BAM and VCF output will be written"
        )

        st.markdown("**MLflow Tracking:**")
        ac1, ac2 = st.columns(2)
        with ac1:
            align_experiment = st.text_input("MLflow Experiment:", value="gwas_variant_calling", key="align_exp")
        with ac2:
            align_run_name = st.text_input("Run Name:", value=f"variant_calling_{_ts}", key="align_run")

        align_btn = st.form_submit_button("Start Alignment Job", type="primary")

    if align_btn:
        if not fastq_r1.strip() or not fastq_r2.strip():
            st.error("Both FASTQ R1 and R2 paths are required.")
        elif not reference_genome.strip():
            st.error("Reference genome path is required.")
        elif not align_experiment.strip() or not align_run_name.strip():
            st.error("MLflow experiment and run name are required.")
        else:
            try:
                with st.spinner("Starting Parabricks alignment job..."):
                    job_run_id = start_parabricks_alignment(
                        user_info=user_info,
                        fastq_r1=fastq_r1.strip(),
                        fastq_r2=fastq_r2.strip(),
                        reference_genome_path=reference_genome.strip(),
                        output_volume_path=output_path.strip(),
                        mlflow_experiment_name=align_experiment.strip(),
                        mlflow_run_name=align_run_name.strip(),
                    )
                    st.success(f"Alignment job started with run id: {job_run_id}")
                    job_id = os.getenv("PARABRICKS_ALIGNMENT_JOB_ID")
                    st.button("View Run", key="align_view_run",
                              on_click=lambda: open_run_window(job_id, job_run_id))
            except Exception as e:
                st.error(f"Failed to start alignment job: {e}")

    # ── Search Past Variant Calling Runs ──
    st.divider()
    st.markdown("###### Search Past Runs:")

    vc_sc1, vc_sc2, vc_sc3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    with vc_sc1:
        vc_search_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                                  selection_mode="single", default="Experiment Name",
                                  key="vc_search_mode")
    with vc_sc2:
        vc_search_text = st.text_input(f"{vc_search_mode} contains:", "gwas", key="vc_search_text")
    with vc_sc3:
        vc_search_btn = st.button("Search", key="vc_search_btn")

    if vc_search_btn:
        with st.spinner("Searching..."):
            st.session_state.pop("vc_run_search_result_df", None)
            st.session_state.pop("selected_vc_run_status", None)

            if vc_search_text.strip():
                if vc_search_mode == "Experiment Name":
                    vc_result_df = search_variant_calling_runs_by_experiment_name(
                        user_email=user_info.user_email, experiment_name=vc_search_text)
                else:
                    vc_result_df = search_variant_calling_runs_by_run_name(
                        user_email=user_info.user_email, run_name=vc_search_text)

                if not vc_result_df.empty:
                    st.session_state["vc_run_search_result_df"] = add_progress_column(vc_result_df, total_steps=2)
                else:
                    st.error("No results found")
            else:
                st.error("Provide a search text")

    if "vc_run_search_result_df" in st.session_state:
        st.divider()
        _show_in_progress_banner(st.session_state["vc_run_search_result_df"])
        vc_view_enabled = (
            "selected_vc_run_status" in st.session_state
            and st.session_state["selected_vc_run_status"] == "alignment_complete"
        )

        vc_v1, vc_v2, vc_v3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with vc_v1:
            vc_view_btn = st.button("View Results", disabled=not vc_view_enabled, key="vc_view_btn")

        vc_selected = st.dataframe(
            st.session_state["vc_run_search_result_df"],
            column_config={"run_id": None},
            use_container_width=True,
            hide_index=True,
            on_select=_set_selected_vc_row_status,
            selection_mode="single-row",
            key="vc_run_search_result_display_df"
        )

        vc_selected_rows = vc_selected.selection.rows
        if len(vc_selected_rows) > 0 and vc_view_btn:
            _display_vc_results_dialog(vc_selected_rows[0])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: GWAS Analysis
# ══════════════════════════════════════════════════════════════════════════════

with gwas_tab:
    st.markdown("### GWAS Analysis with Glow")
    st.markdown(
        "Run genome-wide association analysis using Glow. "
        "Provide a VCF file (from Parabricks output or your own) and a phenotype dataset."
    )

    # ── VCF source selector (Enter Path vs From Variant Calling Run) ──
    vcf_source = st.pills("VCF Source:", ["Enter Path", "From Variant Calling Run"],
                          selection_mode="single", default="Enter Path",
                          key="vcf_source_mode")

    vcf_path_resolved = ""

    if vcf_source == "From Variant Calling Run":
        if "successful_vc_runs_df" not in st.session_state:
            with st.spinner("Loading successful variant calling runs..."):
                successful_vc_runs_df = list_successful_variant_calling_runs(
                    user_email=user_info.user_email)
                st.session_state["successful_vc_runs_df"] = successful_vc_runs_df

        successful_vc_runs_df = st.session_state["successful_vc_runs_df"]

        if successful_vc_runs_df.empty:
            st.warning("No completed variant calling runs found. Use 'Enter Path' instead.")
        else:
            st.markdown("Select a completed variant calling run:")
            vc_picker_selected = st.dataframe(
                successful_vc_runs_df,
                column_config={"run_id": None, "output_vcf": None},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="gwas_vc_picker_df"
            )
            picker_rows = vc_picker_selected.selection.rows
            if len(picker_rows) > 0:
                vcf_path_resolved = successful_vc_runs_df.iloc[picker_rows[0]]["output_vcf"]
                st.caption(f"Selected VCF: `{vcf_path_resolved}`")

        refresh_btn = st.button("Refresh Runs", key="refresh_vc_runs")
        if refresh_btn:
            st.session_state.pop("successful_vc_runs_df", None)
            st.rerun()

    with st.form("gwas_form", enter_to_submit=False):
        if vcf_source == "Enter Path":
            vcf_path_input = st.text_input(
                "VCF file path (UC Volume):",
                value=os.getenv("GWAS_SAMPLE_VCF_PATH", ""),
                placeholder="/Volumes/catalog/schema/gwas_data/alignment/<run_id>/germline.vcf",
                help="Path to VCF file — either from Parabricks output or user-supplied"
            )
        else:
            vcf_path_input = st.text_input(
                "VCF file path (from selected run):",
                value=vcf_path_resolved,
                disabled=True,
                help="Auto-populated from the selected variant calling run above"
            )

        phenotype_path = st.text_input(
            "Phenotype file path (UC Volume, CSV/TSV):",
            value=os.getenv("GWAS_SAMPLE_PHENOTYPE_PATH", ""),
            placeholder="/Volumes/catalog/schema/gwas_data/sample_phenotype/my_phenotypes.csv",
            help="CSV or TSV file with sample IDs and phenotype labels"
        )

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            phenotype_col = st.text_input("Phenotype column:", value="phenotype",
                                          help="Column name containing phenotype labels")
        with gc2:
            contigs = st.text_input("Contigs to analyze:", value="6",
                                    help="Comma-separated list of contigs/chromosomes")
        with gc3:
            hwe_cutoff = st.number_input("HWE cutoff:", value=0.01, format="%.4f",
                                         help="Hardy-Weinberg equilibrium p-value cutoff")

        pvalue_threshold = st.number_input("Firth correction p-value threshold:", value=0.01, format="%.4f")

        st.markdown("**MLflow Tracking:**")
        gc4, gc5 = st.columns(2)
        with gc4:
            gwas_experiment = st.text_input("MLflow Experiment:", value="gwas_analysis", key="gwas_exp")
        with gc5:
            gwas_run_name = st.text_input("Run Name:", value=f"gwas_analysis_{_ts}", key="gwas_run")

        gwas_btn = st.form_submit_button("Start GWAS Analysis", type="primary")

    if gwas_btn:
        final_vcf_path = vcf_path_resolved if vcf_source == "From Variant Calling Run" else vcf_path_input
        if not final_vcf_path.strip():
            st.error("VCF file path is required. Select a variant calling run or enter a path.")
        elif not phenotype_path.strip():
            st.error("Phenotype file path is required.")
        elif not gwas_experiment.strip() or not gwas_run_name.strip():
            st.error("MLflow experiment and run name are required.")
        else:
            try:
                with st.spinner("Starting GWAS analysis job..."):
                    job_run_id = start_gwas_analysis(
                        user_info=user_info,
                        vcf_path=final_vcf_path.strip(),
                        phenotype_path=phenotype_path.strip(),
                        phenotype_column=phenotype_col.strip(),
                        contigs=contigs.strip(),
                        hwe_cutoff=str(hwe_cutoff),
                        pvalue_threshold=str(pvalue_threshold),
                        mlflow_experiment_name=gwas_experiment.strip(),
                        mlflow_run_name=gwas_run_name.strip(),
                    )
                    st.success(f"GWAS analysis job started with run id: {job_run_id}")
                    job_id = os.getenv("GWAS_ANALYSIS_JOB_ID")
                    st.button("View Run", key="gwas_view_run",
                              on_click=lambda: open_run_window(job_id, job_run_id))
            except Exception as e:
                st.error(f"Failed to start GWAS job: {e}")

    # ── Search Past GWAS Runs ──
    st.divider()
    st.markdown("###### Search Past Runs:")

    gwas_sc1, gwas_sc2, gwas_sc3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    with gwas_sc1:
        gwas_search_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                                    selection_mode="single", default="Experiment Name",
                                    key="gwas_search_mode")
    with gwas_sc2:
        gwas_search_text = st.text_input(f"{gwas_search_mode} contains:", "gwas", key="gwas_search_text")
    with gwas_sc3:
        gwas_search_btn = st.button("Search", key="gwas_search_btn")

    if gwas_search_btn:
        with st.spinner("Searching..."):
            st.session_state.pop("gwas_run_search_result_df", None)
            st.session_state.pop("selected_gwas_run_status", None)

            if gwas_search_text.strip():
                if gwas_search_mode == "Experiment Name":
                    result_df = search_gwas_runs_by_experiment_name(
                        user_email=user_info.user_email, experiment_name=gwas_search_text)
                else:
                    result_df = search_gwas_runs_by_run_name(
                        user_email=user_info.user_email, run_name=gwas_search_text)

                if not result_df.empty:
                    st.session_state["gwas_run_search_result_df"] = add_progress_column(result_df, total_steps=3)
                else:
                    st.error("No results found")
            else:
                st.error("Provide a search text")

    if "gwas_run_search_result_df" in st.session_state:
        st.divider()
        _show_in_progress_banner(st.session_state["gwas_run_search_result_df"])
        gwas_view_enabled = (
            "selected_gwas_run_status" in st.session_state
            and st.session_state["selected_gwas_run_status"] == "gwas_complete"
        )

        gwas_v1, gwas_v2, gwas_v3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with gwas_v1:
            gwas_view_btn = st.button("View Results", disabled=not gwas_view_enabled, key="gwas_view_results_btn")

        gwas_selected = st.dataframe(
            st.session_state["gwas_run_search_result_df"],
            column_config={"run_id": None},
            use_container_width=True,
            hide_index=True,
            on_select=_set_selected_gwas_row_status,
            selection_mode="single-row",
            key="gwas_run_search_result_display_df"
        )

        gwas_selected_rows = gwas_selected.selection.rows
        if len(gwas_selected_rows) > 0 and gwas_view_btn:
            _display_gwas_results_dialog(gwas_selected_rows[0])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: VCF Ingestion
# ══════════════════════════════════════════════════════════════════════════════

with ingestion_tab:
    st.markdown("### VCF to Delta Ingestion")
    st.markdown(
        "Convert a VCF file into a queryable Delta table using "
        "Glow. The resulting table can be used "
        "for GWAS, variant annotation, or any downstream analysis."
    )

    # ── VCF source selector ──
    ingest_vcf_source = st.pills("VCF Source:", ["Enter Path", "From Variant Calling Run"],
                                  selection_mode="single", default="Enter Path",
                                  key="ingest_vcf_source_mode")

    ingest_vcf_resolved = ""

    if ingest_vcf_source == "From Variant Calling Run":
        if "ingest_successful_vc_runs_df" not in st.session_state:
            with st.spinner("Loading successful variant calling runs..."):
                ingest_vc_runs_df = list_successful_variant_calling_runs(
                    user_email=user_info.user_email)
                st.session_state["ingest_successful_vc_runs_df"] = ingest_vc_runs_df

        ingest_vc_runs_df = st.session_state["ingest_successful_vc_runs_df"]

        if ingest_vc_runs_df.empty:
            st.warning("No completed variant calling runs found. Use 'Enter Path' instead.")
        else:
            st.markdown("Select a completed variant calling run:")
            ingest_vc_picker = st.dataframe(
                ingest_vc_runs_df,
                column_config={"run_id": None, "output_vcf": None},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="ingest_vc_picker_df"
            )
            picker_rows = ingest_vc_picker.selection.rows
            if len(picker_rows) > 0:
                ingest_vcf_resolved = ingest_vc_runs_df.iloc[picker_rows[0]]["output_vcf"]
                st.caption(f"Selected VCF: `{ingest_vcf_resolved}`")

    with st.form("ingestion_form", enter_to_submit=False):
        if ingest_vcf_source == "Enter Path":
            ingest_vcf_input = st.text_input(
                "VCF file path (UC Volume):",
                placeholder="/Volumes/catalog/schema/volume/variants.vcf",
                help="Path to VCF file in a UC Volume"
            )
        else:
            ingest_vcf_input = st.text_input(
                "VCF file path (from selected run):",
                value=ingest_vcf_resolved,
                disabled=True
            )

        ingest_table_name = st.text_input(
            "Output table name:",
            value=f"vcf_ingested_{_ts}",
            help="Name for the Delta table (will be created in the project catalog/schema)"
        )

        st.markdown("**MLflow Tracking:**")
        ic1, ic2 = st.columns(2)
        with ic1:
            ingest_experiment = st.text_input("MLflow Experiment:", value="vcf_ingestion", key="ingest_exp")
        with ic2:
            ingest_run_name = st.text_input("Run Name:", value=f"vcf_ingestion_{_ts}", key="ingest_run")

        ingest_btn = st.form_submit_button("Start VCF Ingestion", type="primary")

    if ingest_btn:
        final_vcf = ingest_vcf_resolved if ingest_vcf_source == "From Variant Calling Run" else ingest_vcf_input
        if not final_vcf.strip():
            st.error("VCF file path is required.")
        elif not ingest_table_name.strip():
            st.error("Output table name is required.")
        elif not ingest_experiment.strip() or not ingest_run_name.strip():
            st.error("MLflow experiment and run name are required.")
        else:
            try:
                with st.spinner("Starting VCF ingestion job..."):
                    job_run_id = start_vcf_ingestion(
                        user_info=user_info,
                        vcf_path=final_vcf.strip(),
                        output_table_name=ingest_table_name.strip(),
                        mlflow_experiment_name=ingest_experiment.strip(),
                        mlflow_run_name=ingest_run_name.strip(),
                    )
                    st.success(f"VCF ingestion job started with run id: {job_run_id}")
                    job_id = os.getenv("VCF_INGESTION_JOB_ID")
                    st.button("View Run", key="ingest_view_run",
                              on_click=lambda: open_run_window(job_id, job_run_id))
            except Exception as e:
                st.error(f"Failed to start VCF ingestion: {e}")

    # ── Search Past VCF Ingestion Runs ──
    st.divider()
    st.markdown("###### Search Past Runs:")

    ingest_sc1, ingest_sc2, ingest_sc3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    with ingest_sc1:
        ingest_search_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                                       selection_mode="single", default="Experiment Name",
                                       key="ingest_search_mode")
    with ingest_sc2:
        ingest_search_text = st.text_input(f"{ingest_search_mode} contains:", "vcf", key="ingest_search_text")
    with ingest_sc3:
        ingest_search_btn = st.button("Search", key="ingest_search_btn")

    if ingest_search_btn:
        with st.spinner("Searching..."):
            st.session_state.pop("ingest_run_search_result_df", None)

            if ingest_search_text.strip():
                if ingest_search_mode == "Experiment Name":
                    ingest_result_df = search_vcf_ingestion_runs_by_experiment_name(
                        user_email=user_info.user_email, experiment_name=ingest_search_text)
                else:
                    ingest_result_df = search_vcf_ingestion_runs_by_run_name(
                        user_email=user_info.user_email, run_name=ingest_search_text)

                if not ingest_result_df.empty:
                    st.session_state["ingest_run_search_result_df"] = add_progress_column(ingest_result_df, total_steps=2)
                else:
                    st.error("No results found")
            else:
                st.error("Provide a search text")

    if "ingest_run_search_result_df" in st.session_state:
        st.divider()
        _show_in_progress_banner(st.session_state["ingest_run_search_result_df"])
        st.dataframe(
            st.session_state["ingest_run_search_result_df"],
            column_config={"run_id": None},
            use_container_width=True,
            hide_index=True,
            key="ingest_run_search_result_display_df"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Variant Annotation
# ══════════════════════════════════════════════════════════════════════════════


# ── Annotation result-row selection callback ──

def _set_selected_annot_row_status():
    selection = st.session_state["annot_run_search_result_display_df"].selection
    if len(selection["rows"]) > 0:
        idx = selection["rows"][0]
        status = st.session_state["annot_run_search_result_df"].iloc[idx]["status"]
        st.session_state["selected_annot_run_status"] = status
    else:
        st.session_state.pop("selected_annot_run_status", None)


@st.dialog("Variant Annotation Results", width="large")
def _display_annotation_results_dialog(selected_row_index):
    row = st.session_state["annot_run_search_result_df"].iloc[selected_row_index]
    run_id = row["run_id"]
    if isinstance(run_id, pd.Series):
        run_id = run_id.iloc[0]
    run_name = row["run_name"]
    if isinstance(run_name, pd.Series):
        run_name = run_name.iloc[0]

    st.markdown(f"##### Run: {run_name}")

    with st.spinner("Loading annotation results..."):
        try:
            results_df = pull_annotation_results(run_id)

            if results_df.empty:
                st.warning("No pathogenic variants found for this run.")
                return

            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Pathogenic Variants", f"{len(results_df):,}")
            with mc2:
                genes = results_df['gene'].unique().tolist()
                st.metric("Genes", ", ".join(genes))

            st.divider()

            st.markdown("**Pathogenic Variant Details**")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Embedded BRCA Cancer Risk Dashboard
            dashboard_id = os.environ.get("VARIANT_ANNOTATION_DASHBOARD_ID")
            if dashboard_id:
                st.divider()
                st.markdown("**BRCA Cancer Risk Dashboard**")
                host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                if not host_name.startswith("https://"):
                    host_name = "https://" + host_name
                dashboard_url = f"{host_name}/embed/dashboardsv3/{dashboard_id}"
                components.iframe(dashboard_url, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error loading results: {e}")


with annotation_tab:
    st.markdown("### Clinical Variant Annotation")
    st.markdown(
        "Annotate variants with ClinVar "
        "clinical significance data. Filter to specific gene regions, identify "
        "pathogenic variants, and enrich with disease associations."
    )

    # ── Variants table source selector ──
    annot_source = st.pills("Variants Source:", ["Enter Table Name", "From VCF Ingestion"],
                            selection_mode="single", default="Enter Table Name",
                            key="annot_source_mode")

    annot_table_resolved = ""

    if annot_source == "From VCF Ingestion":
        if "successful_ingestion_runs_df" not in st.session_state:
            with st.spinner("Loading successful VCF ingestion runs..."):
                from utils.disease_biology import list_successful_vcf_ingestion_runs
                ingestion_runs_df = list_successful_vcf_ingestion_runs(
                    user_email=user_info.user_email)
                st.session_state["successful_ingestion_runs_df"] = ingestion_runs_df

        ingestion_runs_df = st.session_state["successful_ingestion_runs_df"]

        if ingestion_runs_df.empty:
            st.warning("No completed VCF ingestion runs found. Use 'Enter Table Name' instead.")
        else:
            st.markdown("Select a completed VCF ingestion run:")
            ingestion_picker = st.dataframe(
                ingestion_runs_df,
                column_config={"run_id": None},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="annot_ingestion_picker_df"
            )
            picker_rows = ingestion_picker.selection.rows
            if len(picker_rows) > 0:
                annot_table_resolved = ingestion_runs_df.iloc[picker_rows[0]]["output_table"]
                st.caption(f"Selected table: `{annot_table_resolved}`")

        refresh_ingestion = st.button("Refresh Runs", key="refresh_ingestion_runs")
        if refresh_ingestion:
            st.session_state.pop("successful_ingestion_runs_df", None)
            st.rerun()

    with st.form("annotation_form", enter_to_submit=False):
        if annot_source == "Enter Table Name":
            annot_variants_table = st.text_input(
                "Variants Delta table:",
                placeholder="catalog.schema.my_variants",
                help="Full table name of the Delta table containing variants"
            )
        else:
            annot_variants_table = st.text_input(
                "Variants Delta table (from selected ingestion run):",
                value=annot_table_resolved,
                help="Table from the selected VCF Ingestion run"
            )

        st.markdown("**Gene Regions:**")
        gene_preset = st.selectbox(
            "Preset gene regions:",
            ["BRCA1 + BRCA2", "Custom"],
            help="Select a preset or provide custom gene regions as JSON"
        )

        if gene_preset == "Custom":
            annot_gene_regions = st.text_area(
                "Gene regions JSON:",
                value='[{"name":"BRCA1","contig":"chr17","start":43044292,"end":43170327}]',
                help='JSON array of gene regions with name, contig, start, end'
            )
        else:
            annot_gene_regions = '[{"name":"BRCA1","contig":"chr17","start":43044292,"end":43170327},{"name":"BRCA2","contig":"chr13","start":32315086,"end":32400268}]'
            st.caption("Using BRCA1 (chr17:43,044,292-43,170,327) and BRCA2 (chr13:32,315,086-32,400,268)")

        annot_pathogenic_vcf = st.text_input(
            "Pathogenic VCF path (optional, for demo spiking):",
            placeholder="/Volumes/catalog/schema/variant_annotation_data/demo/brca_pathogenic_corrected.vcf",
            help="Optional: path to a VCF with known pathogenic variants to add to the dataset"
        )

        st.markdown("**MLflow Tracking:**")
        ac1, ac2 = st.columns(2)
        with ac1:
            annot_experiment = st.text_input("MLflow Experiment:", value="variant_annotation", key="annot_exp")
        with ac2:
            annot_run_name = st.text_input("Run Name:", value=f"variant_annotation_{_ts}", key="annot_run")

        annot_btn = st.form_submit_button("Start Variant Annotation", type="primary")

    if annot_btn:
        if not annot_variants_table.strip():
            st.error("Variants table name is required.")
        elif not annot_experiment.strip() or not annot_run_name.strip():
            st.error("MLflow experiment and run name are required.")
        else:
            try:
                with st.spinner("Starting variant annotation job..."):
                    job_run_id = start_variant_annotation(
                        user_info=user_info,
                        variants_table=annot_variants_table.strip(),
                        gene_regions=annot_gene_regions.strip(),
                        pathogenic_vcf_path=annot_pathogenic_vcf.strip() if annot_pathogenic_vcf else "",
                        mlflow_experiment_name=annot_experiment.strip(),
                        mlflow_run_name=annot_run_name.strip(),
                    )
                    st.success(f"Variant annotation job started with run id: {job_run_id}")
                    job_id = os.getenv("VARIANT_ANNOTATION_JOB_ID")
                    st.button("View Run", key="annot_view_run",
                              on_click=lambda: open_run_window(job_id, job_run_id))
            except Exception as e:
                st.error(f"Failed to start annotation job: {e}")

    # ── Search Past Annotation Runs ──
    st.divider()
    st.markdown("###### Search Past Runs:")

    annot_sc1, annot_sc2, annot_sc3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    with annot_sc1:
        annot_search_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                                      selection_mode="single", default="Experiment Name",
                                      key="annot_search_mode")
    with annot_sc2:
        annot_search_text = st.text_input(f"{annot_search_mode} contains:", "variant", key="annot_search_text")
    with annot_sc3:
        annot_search_btn = st.button("Search", key="annot_search_btn")

    if annot_search_btn:
        with st.spinner("Searching..."):
            st.session_state.pop("annot_run_search_result_df", None)
            st.session_state.pop("selected_annot_run_status", None)

            if annot_search_text.strip():
                if annot_search_mode == "Experiment Name":
                    annot_result_df = search_variant_annotation_runs_by_experiment_name(
                        user_email=user_info.user_email, experiment_name=annot_search_text)
                else:
                    annot_result_df = search_variant_annotation_runs_by_run_name(
                        user_email=user_info.user_email, run_name=annot_search_text)

                if not annot_result_df.empty:
                    st.session_state["annot_run_search_result_df"] = add_progress_column(annot_result_df, total_steps=3)
                else:
                    st.error("No results found")
            else:
                st.error("Provide a search text")

    if "annot_run_search_result_df" in st.session_state:
        st.divider()
        _show_in_progress_banner(st.session_state["annot_run_search_result_df"])
        annot_view_enabled = (
            "selected_annot_run_status" in st.session_state
            and st.session_state["selected_annot_run_status"] == "annotation_complete"
        )

        annot_v1, annot_v2, annot_v3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with annot_v1:
            annot_view_btn = st.button("View Results", disabled=not annot_view_enabled, key="annot_view_results_btn")

        annot_selected = st.dataframe(
            st.session_state["annot_run_search_result_df"],
            column_config={"run_id": None},
            use_container_width=True,
            hide_index=True,
            on_select=_set_selected_annot_row_status,
            selection_mode="single-row",
            key="annot_run_search_result_display_df"
        )

        annot_selected_rows = annot_selected.selection.rows
        if len(annot_selected_rows) > 0 and annot_view_btn:
            _display_annotation_results_dialog(annot_selected_rows[0])

