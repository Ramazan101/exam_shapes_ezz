import streamlit as st, requests

st.title("Image Classification")
file = st.file_uploader("label",type=["jpg", "png", "jpeg"])

if file and st.button("Button"):
    st.image(file)
    response = requests.post("http://127.0.0.1:8000/predict", files={"file": file})
    if response.ok:
        st.success(f"result: {response.json()['class'].title()}")
    else:
        st.error("server error")