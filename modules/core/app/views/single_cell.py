
import streamlit as st
import pandas as pd

from scripts import single_cell_helper as sch
from genesis_workbench.models import get_available_models, get_deployed_models
from genesis_workbench.workbench import execute_query

available_models_df = get_available_models()
available_models_df["model_labels"] = (available_models_df["model_id"].astype(str) + " - " 
                                       + available_models_df["model_display_name"].astype(str) + " [ " 
                                       + available_models_df["model_uc_name"].astype(str) + " v"
                                       + available_models_df["model_uc_version"].astype(str) + " ]"
                                       )

deployed_models_df = get_deployed_models()
deployed_models_df.columns = ["Id", "Name", "Source Version", "UC Name", "UC Version", "Deployed Date"]

st.title(":material/microbiology:  Single Cell Studies")

settings_tab, embeddings_tab = st.tabs(["Settings","Embeddings"])

with settings_tab:
    sch.display_settings_tab({})

with embeddings_tab:
    sch.display_embeddings_tab({})