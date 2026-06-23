import streamlit as st
import runpy

st.set_page_config(
    page_title="Multilingual Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Multilingual Fake News Detection")

language = st.selectbox(
    "Select Language",
    ["English", "Tamil"]
)

st.markdown("---")

if language == "English":
    runpy.run_path("ENGapp.py")

elif language == "Tamil":
    runpy.run_path("TAMapp.py")