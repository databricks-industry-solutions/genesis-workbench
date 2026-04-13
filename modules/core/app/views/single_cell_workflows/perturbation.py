"""Single Cell — Gene Perturbation Prediction tab (placeholder)."""

import streamlit as st


def render():
    st.markdown("##### Gene Perturbation Prediction")
    st.markdown(
        "Predict the effect of gene knockouts or overexpression on cell state "
        "using scGPT's masked language modeling capability."
    )
    st.info(
        "This workflow requires an extended scGPT model registration that supports "
        "perturbation prediction (masked inference). The model variant `scgpt_perturbation` "
        "needs to be registered and deployed before this feature can be used.\n\n"
        "**Coming soon** — once the scGPT perturbation model is registered, this tab will allow you to:\n"
        "- Select a cell type / cluster from a completed analysis\n"
        "- Specify a gene to knock out or overexpress\n"
        "- View predicted expression changes and affected pathways"
    )
