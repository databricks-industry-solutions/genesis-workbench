import streamlit as st

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models)

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
        deployed_single_cell_models_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        deployed_single_cell_models_df.columns = ["Model Id", "Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]

settings_tab, processing_tab = st.tabs([
    "Settings",
    "Raw Single Cell Processing",
])

with settings_tab:
    settings.render(available_single_cell_models_df, deployed_single_cell_models_df)

with processing_tab:
    processing.render()
