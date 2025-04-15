import streamlit as st


st.title(":material/home: Home")

col1,col2,col3,col4 = st.columns([1,8,1,1])

with col2:
    st.text_input("Search:","", placeholder="Search Documentation", label_visibility="collapsed")

with col3:
    st.button("Search")

col1,col2,col3 = st.columns([1,1,1])
with col2:
    st.pills("Actions",[":material/schedule: Recent",
                        ":material/favorite: Favorites",
                        ":material/insert_chart: Popular",
                        ":material/featured_seasonal_and_gifts: What's New"] ,
             label_visibility="collapsed",
             default=":material/schedule: Recent",
             selection_mode="single")
    

