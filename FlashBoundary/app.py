
import streamlit as st
st.title("FlashBoundary AI")
st.write("Upload MP4 to generate cricket highlights.")
video = st.file_uploader("Upload Match Video", type=["mp4"])
if video:
    st.success("Video uploaded successfully.")
