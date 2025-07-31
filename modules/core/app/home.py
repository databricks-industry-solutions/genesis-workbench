import streamlit as st
import os
from genesis_workbench.workbench import initialize
from utils.streamlit_helper import get_user_info  
from databricks.sdk import WorkspaceClient
from genesis_workbench.workbench import get_user_settings, save_user_settings

st.set_page_config(layout="wide")
#delete the top right menu

with st.spinner("Initializing"):
    initialize(
        core_catalog_name = os.environ["CORE_CATALOG_NAME"],
        core_schema_name = os.environ["CORE_SCHEMA_NAME"],
        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]
    )
    user_info = get_user_info()
    if "user_settings" not in st.session_state:        
        user_settings = get_user_settings(user_email=user_info.user_email)
        st.session_state["user_settings"] = user_settings
    
    user_settings = st.session_state["user_settings"]

st.logo("images/blank.png", size="large", icon_image="images/dbx_logo_icon_2.png")
st.sidebar.image("images/dbx_logo_1.png", width=200)

home_page = st.Page(
    page="views/home.py",
    title="Home",
    icon=":material/home:",
    default=True
)

single_cell_page = st.Page(
    page="views/single_cell.py",
    title="Single Cell",
    icon=":material/microbiology:"
)

protein_page = st.Page(
    page="views/protein_studies.py",
    title="Protein Studies",
    icon=":material/biotech:"
)

profile_page = st.Page(
    page="views/user_profile.py",
    title=f"Profile { '' if 'setup_done' in user_settings and user_settings['setup_done']=='Y' else '⚠️ Setup Incomplete'}",
    icon=":material/account_circle:"
)

# small_molecules_page = st.Page(
#     page="views/small_molecules.py",
#     title="Small Molecules",
#     icon=":material/vaccines:"
# )

settings_page = st.Page(
    page="views/settings.py",
    title="Settings",
    icon=":material/settings:"
)

monitoring_alerts_page = st.Page(
    page="views/monitoring_alerts.py",
    title="Monitoring and Alerts",
    icon=":material/monitoring:"
)

bionemo_esm_page = st.Page(
    page="views/bionemo/bionemo_esm.py",
    title="NVIDIA BioNeMo©",
    icon=":material/genetics:"
)

# bionemo_geneformer_page = st.Page(
#     page="views/bionemo/bionemo_geneformer.py",
#     title="[NVIDIA BioNeMo©] Geneformer",
#     icon=":material/genetics:"
# )

menu_pages = {
    f"{user_settings['user_display_name'] if 'user_display_name' in user_settings else user_info.user_display_name}":[
        profile_page,
    ],
    "Workbench": [
        home_page,
        single_cell_page,
        protein_page,        
        # small_molecules_page,
        bionemo_esm_page,        
        # bionemo_geneformer_page
    ],
    "Management" : [
        monitoring_alerts_page,
        settings_page    
    ]    

}

pg = st.navigation(pages=menu_pages)
pg.run()


