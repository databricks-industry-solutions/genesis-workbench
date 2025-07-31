import streamlit as st
from utils.streamlit_helper import get_user_info  
from genesis_workbench.workbench import get_user_settings

st.title(":material/home: Home")

with st.spinner("Getting user information"):
    user_info = get_user_info()
    user_settings = get_user_settings(user_email=user_info.user_email)


col1,col2,col3,col4 = st.columns([1,8,1,1])

with col2:
    st.text_input("Search:","", placeholder="Search Documentation [Coming Soon]", label_visibility="collapsed")

with col3:
    st.button("Search")

col1,col2,col3 = st.columns([1,3,1])
with col2:

    if "setup_done" in user_settings and user_settings["setup_done"]=="Y":
        
        st.pills("Actions",[":material/schedule: Recent",
                            ":material/favorite: Favorites",
                            ":material/insert_chart: Popular",
                            ":material/featured_seasonal_and_gifts: What's New"] ,
                label_visibility="collapsed",
                default=":material/schedule: Recent",
                selection_mode="single")
        
    else:
        st.markdown("## Welcome to Genesis Workbench.")
        st.markdown("#### ⚠️ Your profile setup is incomplete.")
        st.write("Finish your profile setup using Profile tab.")
    

