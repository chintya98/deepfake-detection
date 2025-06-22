import streamlit as st

st.title("Deepfake Detection App")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)
    # proses model dan tampilkan prediksi nanti di sini
