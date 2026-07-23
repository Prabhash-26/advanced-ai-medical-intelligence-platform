from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = st.sidebar.text_input("API URL", value=os.getenv("API_URL", "http://localhost:8000/api/v1"))

st.set_page_config(page_title="Medical AI Intelligence", layout="wide")
st.title("Advanced AI Medical Intelligence Platform")
st.caption("Academic decision-support demo. Not for clinical diagnosis.")

uploaded = st.file_uploader("Upload a chest X-ray or medical image", type=["png", "jpg", "jpeg"])

if uploaded and st.button("Analyze Image", type="primary"):
    with st.spinner("Running model, Grad-CAM, and report generation..."):
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=120)
    if response.ok:
        data = response.json()
        left, right = st.columns([1, 1])
        with left:
            st.image(uploaded.getvalue(), caption="Uploaded image", use_container_width=True)
            st.metric("Prediction", data["predicted_class"], f"{data['confidence']:.1%} confidence")
            st.bar_chart(data["probabilities"])
        with right:
            if data.get("heatmap_url"):
                st.image(f"{API_URL}{data['heatmap_url'].replace('/api/v1', '')}", caption="Grad-CAM overlay")
            st.subheader("AI-assisted report")
            st.write(data["report"])
    else:
        st.error(response.text)

st.divider()
st.subheader("Prediction History")
try:
    history = requests.get(f"{API_URL}/history", timeout=20).json()
    st.dataframe(history, use_container_width=True)
except Exception:
    st.info("Start the FastAPI backend to load history.")
