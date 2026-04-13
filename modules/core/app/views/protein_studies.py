import streamlit as st

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models,
                                      get_batch_models)

import pandas as pd
from utils.streamlit_helper import get_user_info

from views.protein_studies_workflows import settings, structure_prediction, protein_design, sequence_search, inverse_folding

st.title(":material/biotech: Protein Studies")

with st.spinner("Loading data"):
    if "available_protein_models_df" not in st.session_state:
        available_protein_models_df = get_available_models(ModelCategory.PROTEIN_STUDIES)
        available_protein_models_df["model_labels"] = (available_protein_models_df["model_id"].astype(str) + " - "
                                            + available_protein_models_df["model_display_name"].astype(str) + " [ "
                                            + available_protein_models_df["model_uc_name"].astype(str) + " v"
                                            + available_protein_models_df["model_uc_version"].astype(str) + " ]"
                                            )
        st.session_state["available_protein_models_df"] = available_protein_models_df
    available_protein_models_df = st.session_state["available_protein_models_df"]

    if "deployed_protein_models_df" not in st.session_state:
        rt_df = get_deployed_models(ModelCategory.PROTEIN_STUDIES)
        rt_df.columns = ["Model Id", "Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        rt_df["Type"] = "Real-time"
        rt_df["Cluster"] = ""
        rows = [rt_df]
        try:
            batch_df = get_batch_models("protein_studies")
            if not batch_df.empty:
                batch_df.columns = ["Name", "Description", "Endpoint Name", "Cluster"]
                batch_df["Type"] = "Batch"
                batch_df["Model Id"] = None
                batch_df["Deploy Id"] = None
                batch_df["Model Name"] = None
                batch_df["Source Version"] = None
                batch_df["UC Name/Version"] = None
                rows.append(batch_df)
        except Exception:
            pass
        deployed_protein_models_df = pd.concat(rows, ignore_index=True)
        for col in ["Model Id", "Deploy Id"]:
            if col in deployed_protein_models_df.columns:
                deployed_protein_models_df[col] = deployed_protein_models_df[col].astype(str).replace("None", "")
        st.session_state["deployed_protein_models_df"] = deployed_protein_models_df
    deployed_protein_models_df = st.session_state["deployed_protein_models_df"]

user_info = get_user_info()

settings_tab, sequence_search_tab, protein_structure_prediction_tab, protein_design_tab, inverse_folding_tab = st.tabs(["Deployed Models", "Sequence Search", "Protein Structure Prediction", "Protein Design", "Inverse Folding"])

with settings_tab:
    settings.render(available_protein_models_df, deployed_protein_models_df)

with sequence_search_tab:
    sequence_search.render()

with protein_structure_prediction_tab:
    structure_prediction.render()

with protein_design_tab:
    protein_design.render()

with inverse_folding_tab:
    inverse_folding.render()
