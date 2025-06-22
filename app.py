import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from utils.gradcam import make_gradcam_heatmap, display_gradcam
from PIL import Image

@st.cache_resource
def load_model():
  return tf.keras.models.load_model("model/model-with-new-dataset.h5")

model = load_model()

st.set_page_config(layout="wide")
st.title("Deepfake Image Detection")

st.subheader("Input")
container = st.container(border=True)
with container:
  col1, col2 = st.columns(2)
  with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jfif"])
    image_pil = None
    img_preprocessed = None
    if uploaded_file is not None:
      image_pil = Image.open(uploaded_file).convert("RGB")
      st.image(image_pil, caption='Uploaded Image', width=300)

      # Preprocess
      img_resized = image_pil.resize((299, 299))
      img_array = image.img_to_array(img_resized)
      img_array_expanded = np.expand_dims(img_array, axis=0)
      img_preprocessed = preprocess_input(img_array_expanded)

  with col2:
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
    predict_clicked = False
    if uploaded_file and st.button("Predict", use_container_width=True):
      predict_clicked = True

if predict_clicked and img_preprocessed is not None:
  st.subheader("Output")
  result_container = st.container(border=True)
  with result_container:
    prediction = model.predict(img_preprocessed)
    prob = prediction[0][0]
    label = "real" if prob > threshold else "fake"

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Heatmap (Grad-CAM)")
      heatmap = make_gradcam_heatmap(
        img_preprocessed,
        model,
        last_conv_layer_name="block14_sepconv2"
      )
      superimposed_img = display_gradcam(np.array(img_resized), heatmap)
      st.image(superimposed_img, caption='Grad-CAM Heatmap', width=300)

    with col2:
      st.subheader("Prediction Result")
      st.markdown(f"**Label:** {label.upper()}")
      st.markdown(f"**Confidence Score:** {prob:.4f}")

