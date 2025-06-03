import streamlit as st

st.set_page_config(layout="wide")
#delete the top right menu

st.logo("images/blank.png", size="large", icon_image="images/dbx_logo_icon_2.png")
st.sidebar.image("images/big_blank.png", width=200)
st.sidebar.image("images/dbx_logo_1.png", width=200)

home_page = st.Page(
    page="views/home.py",
    title="Home",
    icon=":material/home:"
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

small_molecules_page = st.Page(
    page="views/small_molecules.py",
    title="Small Molecules",
    icon=":material/vaccines:"
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
    page="views/bionemo/bionemo_esm.py",
    title="[NVIDIA BioNeMo©] ESMFold2",
    icon=":material/genetics:"
)

bionemo_geneformer_page = st.Page(
    page="views/bionemo/bionemo_geneformer.py",
    title="[NVIDIA BioNeMo©] Geneformer",
    icon=":material/genetics:"
)

menu_pages = {
    "Workbench": [
        home_page,
        single_cell_page,
        protein_page,
        small_molecules_page,
        bionemo_esm_page,
        bionemo_geneformer_page
    ],
    "Management" : [
        monitoring_alerts_page,
        settings_page    
    ]    

}

pg = st.navigation(pages=menu_pages)



pg.run()
