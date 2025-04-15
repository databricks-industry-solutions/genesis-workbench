import streamlit as st
import pandas as pd

def display_settings_tab(data:dict):
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("###### Deployed Models")
    with col2:
        c1,c2,c3 = st.columns([1,1,1])
        with c1:
            st.button("Permissions")    
        with c2:
            st.button("Delete")

    df = pd.DataFrame([
        {"Model": "Teddy v1.0.3", "Deploy Date": "Apr 24, 2025"},
        {"Model": "SCimilarity v0.0.3", "Deploy Date": "Feb 11, 2025"},
        {"Model": "Geneformer v1.2.3", "Deploy Date": "Apr 24, 2025"},
    ])

    st.dataframe(df, 
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row")
   
    st.markdown("###### Available Models:")
    
    p1,p2 = st.columns([2,1])

    with p1:
        col1, col2, = st.columns([1,1])    
        with col1:
            select_models = st.selectbox("Model:",["scGPT","Nicheformer"],label_visibility="collapsed",)

        with col2:
            deploy_button = st.button('Deploy',key="btn_deploy")

        st.markdown("###### Import Models:")
        col1, col2, = st.columns([1,1], vertical_alignment="bottom")    
        with col1:
            select_models = st.selectbox("Source:",["Unity Catalog","Hugging Face","PyPi"],label_visibility="visible")

        with col2:
            import_button = st.button('Import',key="btn_import")


def display_embeddings_tab(data:dict):
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("###### Generate Embeddings")
    with col2:        
        st.button("View Past Runs")    

    st.write("Select Models:")
    df = pd.DataFrame([
        {"Model": "Teddy v1.0.3", "Deploy Date": "Apr 24, 2025"},
        {"Model": "SCimilarity v0.0.3", "Deploy Date": "Feb 11, 2025"},
        {"Model": "Geneformer v1.2.3", "Deploy Date": "Apr 24, 2025"},
    ])

    st.dataframe(df, 
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row")
    
    st.write("NOTE: A result table will be created for EACH model selected.")

    col1, col2, col3 = st.columns([1,1,1], vertical_alignment="bottom")
    with col1:        
        st.text_input("Data Location:","")
        st.text_input("Result Schema Name:","")
        st.text_input("Result Table Prefix:","")
    
    with col2:
        st.write("")
        st.toggle("Perform Evaluation?")            
        st.text_input("Ground Truth Data Location:","")
        st.text_input("MLflow Experiment Name:","")
    
    st.button("Generate Embeddings")





