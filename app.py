from utils.similarity import sentence_similarity_analysis
from utils.text_analysis import analyze_text
import streamlit as st
from models.plagiarism_model import detect_plagiarism
from models.ai_text_detector import detect_ai_text
from utils.file_reader import extract_text

st.title("Plagiarism and AI content Detector")

uploaded_file = st.file_uploader("Choose a file", type=["txt","pdf","docx"])
if uploaded_file is not None:
    text = extract_text(uploaded_file)
    analysis = analyze_text(text)

    st.subheader("EXTRACTED TEXT")
    st.write(text)

    plagiarism_score = detect_plagiarism(text)
    ai_score , ai_label = detect_ai_text(text)

    st.subheader("📊 Analysis Results")



    st.write(f" 📑 Plagiarism score: {plagiarism_score:.2f}%")
    st.progress(int(plagiarism_score))

    st.write(f"🤖 AI score: {ai_score:.2f}%")
    st.progress(ai_score/100)
    if ai_label == "AI":
        st.error("🤖 AI generated content")
    else:
        st.success(f"👦👩 human written content")

    st.subheader("📊 Text Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Words", analysis["word_count"])
        st.metric("Sentences", analysis["sentence_count"])
        st.metric("Reading Time", f"{analysis['reading_time']} min")

    with col2:
        st.metric("Avg Sentence Length", analysis["avg_sentence_length"])
        st.metric("Lexical Diversity", analysis["lexical_diversity"])

    st.subheader("🧠 AI Indicators")

    if analysis["lexical_diversity"] < 0.45:
        st.warning("⚠️ Low lexical diversity")

    if analysis["avg_sentence_length"] > 20:
        st.warning("⚠️ Long and uniform sentence structure")

    if ai_score > 90:
        st.error("🔴 Very high AI confidence")

    elif ai_score > 70:
        st.warning("🟠 Moderate AI confidence")

    else:
        st.success("🟢 Mostly human-like writing")

    st.subheader("📑 Sentence-wise Plagiarism")

    results = sentence_similarity_analysis(text)



    for item in results:

        similarity = float(item["similarity"])

        sentence = item["sentence"]

        if similarity > 70:
            st.error(f"🔴 {similarity:.2f}%  {sentence}")

        elif similarity > 40:
            st.warning(f"🟡 {similarity:.2f}%  {sentence}")

        else:
            st.success(f"🟢 {similarity:.2f}%  {sentence}")


