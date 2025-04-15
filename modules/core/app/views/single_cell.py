
import streamlit as st
import pandas as pd

from scripts import single_cell_helper as sch

st.title(":material/microbiology:  Single Cell Studies")

settings_tab, embeddings_tab = st.tabs(["Settings","Embeddings"])

with settings_tab:
    sch.display_settings_tab({})

with embeddings_tab:
    sch.display_embeddings_tab({})