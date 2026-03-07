# ==========================================
# Fish Image Classification Streamlit App
# ==========================================

# ---------- Import Libraries ----------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- Page Setup ----------
st.set_page_config(page_title="Fish Image Classification", page_icon="🐟")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/cnn_model.keras")
    return model

model = load_model()

# ---------- Class Names ----------
class_names = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# ---------- Title ----------
st.title("🐟 Fish Image Classification App")
st.write("Upload a fish image and the model will predict the fish species.")

# ---------- Upload Image ----------
uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg", "jpeg", "png"])

# ---------- If Image Uploaded ----------
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=400)

    # ---------- Preprocess Image ----------
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------- Prediction ----------
    predictions = model.predict(img_array)[0]

    best_index = np.argmax(predictions)
    best_class = class_names[best_index]
    confidence = predictions[best_index]

    threshold = 0.50

    st.subheader("Prediction")

    if confidence < threshold or best_class == "animal fish":
        st.warning("⚠️ Species not available in our dataset.")
    else:
        st.success(f"🐟 Predicted Fish Species: {best_class}")
        st.write(f"Confidence Score: {confidence:.2f}")

    