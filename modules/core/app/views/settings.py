import streamlit as st
import os

st.title(":material/settings: Settings")

general_tab, access_tab = st.tabs(["General","Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])    

    with col1:
        st.text_input("Application Schema Location: ", "main.dbx_genesis_wb")
        
        st.write(f"SQL Warehouse Host Name:{os.getenv('SQL_WAREHOUSE','NONE')}")