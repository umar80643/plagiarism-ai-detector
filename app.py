"""Streamlit UI for the plagiarism / AI-content detector.

This is a thin client: all detection logic lives behind the FastAPI backend
(api/main.py), not in this file. That means the same detection logic is also
usable from a script, another service, or a mobile app -- not just from
inside this Streamlit process. Run the API first:

    uvicorn api.main:app --reload

then:

    streamlit run app.py
"""
from __future__ import annotations
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "dev-key")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Plagiarism & AI Content Detector", layout="wide")
st.title("Plagiarism and AI Content Detector")


def _api_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _api_post(path: str, **kwargs) -> dict | None:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", headers=HEADERS, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


health = _api_get("/v1/health")
if health is None:
    st.error(f"Can't reach the API at {API_BASE_URL}. Start it with `uvicorn api.main:app --reload`.")
    st.stop()
st.caption(
    f"API: {API_BASE_URL} · corpus documents: {health['plagiarism_index_documents']} "
    f"· AI detector: {health['ai_detector_mode']}"
)

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    plagiarism = _api_post(
        "/v1/detect/plagiarism/file",
        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
    )
    # The file endpoint doesn't return the extracted text, so re-read the
    # small text preview client-side only for .txt; for pdf/docx just show a
    # note instead of re-implementing extraction here.
    if uploaded_file.name.lower().endswith(".txt"):
        st.subheader("Extracted text")
        st.write(uploaded_file.getvalue().decode("utf-8", errors="replace"))

    if plagiarism is None:
        st.stop()

    st.subheader("📊 Analysis results")
    st.write(f"📑 Plagiarism score (best match in reference corpus): {plagiarism['score']:.2f}%")
    st.progress(int(plagiarism["score"]))
    if plagiarism["matches"]:
        with st.expander("Matched sources"):
            for match in plagiarism["matches"]:
                st.write(f"- {match['source']}: {match['similarity']:.2f}%")
    else:
        st.caption("No reference documents found — add some to data/corpus/ and run build_index.py.")

    if uploaded_file.name.lower().endswith(".txt"):
        ai_result = _api_post(
            "/v1/detect/ai-text", json={"text": uploaded_file.getvalue().decode("utf-8", errors="replace")}
        )
        if ai_result:
            st.write(f"🤖 AI score: {ai_result['score']:.2f}%")
            st.progress(ai_result["score"] / 100)
            if health["ai_detector_mode"] == "heuristic":
                st.caption("Using heuristic scoring. Run train_ai_detector.py for a trained classifier instead.")
            if ai_result.get("explanation"):
                with st.expander("Why?"):
                    for reason in ai_result["explanation"]:
                        st.write(f"- {reason}")
            if ai_result["label"] == "AI":
                st.error("🤖 Likely AI-generated content")
            else:
                st.success("🙂 Likely human-written content")

    st.subheader("📑 Sentence matches against the reference corpus")
    st.caption("Looks OUTSIDE this document — this is the actual plagiarism check.")
    for item in plagiarism["sentence_matches"]:
        if item["source"] is None:
            continue
        similarity = item["similarity"]
        label = f"{similarity:.2f}%  \"{item['sentence']}\"  ↔  {item['source']}: \"{item['matched_sentence']}\""
        if similarity > 70:
            st.error(f"🔴 {label}")
        elif similarity > 40:
            st.warning(f"🟡 {label}")

st.divider()
st.caption("Paste text directly (skips file upload) for a quick check:")
pasted_text = st.text_area("Text to analyse", height=120)
if st.button("Analyse pasted text", disabled=not pasted_text.strip()):
    plagiarism = _api_post("/v1/detect/plagiarism", json={"text": pasted_text})
    ai_result = _api_post("/v1/detect/ai-text", json={"text": pasted_text})
    if plagiarism:
        st.write(f"📑 Plagiarism score: {plagiarism['score']:.2f}%")
    if ai_result:
        st.write(f"🤖 AI score: {ai_result['score']:.2f}% ({ai_result['label']})")
