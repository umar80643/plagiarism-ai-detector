import docx
import streamlit as st
from models.plagiarism_model import detect_plagiarism
from utils.file_reader import read_text , read_pdf ,read_docx

st.title("Plagiarism Detector")

uploaded_file = st.file_uploader("Upload a file",type=["txt","pdf","docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        text = read_text(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text = read_docx(uploaded_file)
    score = detect_plagiarism(text)
    st.write(f"palgiarism score: {score}%")