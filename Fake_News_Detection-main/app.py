import streamlit as st
import runpy
import os

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if language == "English":
    runpy.run_path(os.path.join(BASE_DIR, "ENGapp.py"))

elif language == "Tamil":
    runpy.run_path(os.path.join(BASE_DIR, "TAMapp.py"))
