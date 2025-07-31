import streamlit as st
import os
import base64
from utils.streamlit_helper import get_user_info

st.title(":material/settings: Settings")

general_tab, access_tab = st.tabs(["General","Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])    

    with col1:
        
        core_catalog_name = os.environ["CORE_CATALOG_NAME"]
        core_schema_name = os.environ["CORE_SCHEMA_NAME"]
        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]

        st.text_input("Application Schema Location: ", f"{core_catalog_name}.{core_schema_name}")
        
        st.write(f"SQL Warehouse Host Name:{sql_warehouse_id}")

