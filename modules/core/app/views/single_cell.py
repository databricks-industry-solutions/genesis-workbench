import streamlit as st
import pandas as pd

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models,
                                      get_batch_models)

from views.single_cell_workflows import settings, processing

st.title(":material/microbiology:  Single Cell Studies")

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
        rt_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        rt_df.columns = ["Model Id", "Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        rt_df["Type"] = "Real-time"
        rt_df["Cluster"] = ""
        rows = [rt_df]
        try:
            batch_df = get_batch_models("single_cell")
            if not batch_df.empty:
                batch_df.columns = ["Name", "Description", "Endpoint Name", "Cluster"]
                batch_df["Type"] = "Batch"
                batch_df["Model Id"] = ""
                batch_df["Deploy Id"] = ""
                batch_df["Model Name"] = ""
                batch_df["Source Version"] = ""
                batch_df["UC Name/Version"] = ""
                rows.append(batch_df)
        except Exception:
            pass
        deployed_single_cell_models_df = pd.concat(rows, ignore_index=True)
        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]

settings_tab, processing_tab = st.tabs([
    "Deployed Models",
    "Raw Single Cell Processing",
])

with settings_tab:
    settings.render(available_single_cell_models_df, deployed_single_cell_models_df)

with processing_tab:
    processing.render()
