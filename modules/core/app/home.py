import streamlit as st
import os
import glob
from genesis_workbench.workbench import initialize
from utils.streamlit_helper import get_user_info
from databricks.sdk import WorkspaceClient
from genesis_workbench.workbench import get_user_settings, get_deployed_modules

st.set_page_config(layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)
deployed_modules = []

with st.spinner("Initializing"):
    if "system_settings_initialized" not in st.session_state:
        initialize(
            core_catalog_name = os.environ["CORE_CATALOG_NAME"],
            core_schema_name = os.environ["CORE_SCHEMA_NAME"],
            sql_warehouse_id = os.environ["SQL_WAREHOUSE"]
        )
        st.session_state["system_settings_initialized"] = "true"
        deployed_modules = get_deployed_modules()
       
        st.session_state["deployed_modules"] = deployed_modules

        # Build documentation index
        doc_index = []
        doc_dir = os.path.join(os.path.dirname(__file__), "documentation")
        for md_path in sorted(glob.glob(os.path.join(doc_dir, "*.md"))):
            if os.path.basename(md_path) == "index.md":
                continue
            with open(md_path, "r") as f:
                content = f.read()
            title = os.path.basename(md_path).replace(".md", "").replace("_", " ").title()
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
            doc_index.append({"title": title, "content": content, "file": os.path.basename(md_path)})
        st.session_state["doc_index"] = doc_index

    deployed_modules = st.session_state["deployed_modules"]
    
    user_info = get_user_info()
    if "user_settings" not in st.session_state:        
        user_settings = get_user_settings(user_email=user_info.user_email)
        st.session_state["user_settings"] = user_settings
    
    user_settings = st.session_state["user_settings"]


st.logo("images/blank.png", size="large", icon_image="images/dbx_logo_1.png")
st.sidebar.image("images/dbx_logo_1.png", width=600)

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

small_molecules_page = st.Page(
    page="views/small_molecules.py",
    title="Small Molecules",
    icon=":material/science:"
)

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
    page="views/nvidia/bionemo_esm.py",
    title="NVIDIA BioNeMo©",
    icon=":material/genetics:"
)

parabricks_page = st.Page(
    page="views/nvidia/parabricks.py",
    title="NVIDIA Parabricks©",
    icon=":material/genetics:"
)

disease_biology_page = st.Page(
    page="views/disease_biology.py",
    title="Disease Biology",
    icon=":material/coronavirus:"
)

# bionemo_geneformer_page = st.Page(
#     page="views/bionemo/bionemo_geneformer.py",
#     title="[NVIDIA BioNeMo©] Geneformer",
#     icon=":material/genetics:"
# )

workbench_pages = [home_page]
if "single_cell" in deployed_modules:
    workbench_pages.append(single_cell_page)
if "protein_studies" in deployed_modules:
    workbench_pages.append(protein_page)
if "small_molecule" in deployed_modules:
    workbench_pages.append(small_molecules_page)
if "disease_biology" in "disease_biology": #deployed_modules:
    workbench_pages.append(disease_biology_page)
if "bionemo" in deployed_modules:
    workbench_pages.append(bionemo_esm_page)
if "parabricks" in deployed_modules:
    workbench_pages.append(parabricks_page)


menu_pages = {
    f"{user_settings['user_display_name'] if 'user_display_name' in user_settings else user_info.user_display_name}":[
        profile_page,
    ],
    "Workbench": workbench_pages,
    "Management" : [
        monitoring_alerts_page,
        settings_page    
    ]    

}

pg = st.navigation(pages=menu_pages)
pg.run()


