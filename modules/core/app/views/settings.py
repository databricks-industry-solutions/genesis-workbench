import streamlit as st
import os
from genesis_workbench.workbench import get_app_context

st.title(":material/settings: Settings")

general_tab, access_tab = st.tabs(["General","Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])    

    with col1:
        
        ctx = get_app_context()

        st.text_input("Application Schema Location: ", f"{ctx.core_catalog_name}.{ctx.core_schema_name}")
        
        st.write(f"SQL Warehouse Host Name:{os.getenv('SQL_WAREHOUSE','NONE')}")