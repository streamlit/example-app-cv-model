import pandas as pd
import requests
import streamlit as st
from models import SUPPORTED_MODELS, bytes_to_array, prepare_image

st.set_page_config(layout="wide")
st.title(":camera: Computer vision app")


# Let user upload a picture
with st.sidebar:
    st.title("Upload a picture")

    upload_type = st.radio(
        label="How to upload the picture",
        options=(("From file", "From URL", "From webcam")),
    )

    image_bytes = None

    if upload_type == "From file":
        file = st.file_uploader(
            "Upload image file", type=[".png"], accept_multiple_files=False
        )
        if file:
            image_bytes = file.getvalue()

    if upload_type == "From URL":
        url = st.text_input("Paste URL")
        if url:
            image_bytes = requests.get(url).content

    if upload_type == "From webcam":
        camera = st.camera_input("Take a picture!")
        if camera:
            image_bytes = camera.getvalue()

st.write("## Uploaded picture")
if image_bytes:
    st.write("🎉 Here's what you uploaded!")
    st.image(image_bytes, width=200)
else:
    st.warning("👈 Please upload an image first...")
    st.stop()


st.write("## Model prediction")

# model_name = st.selectbox("Choose model", SUPPORTED_MODELS.keys())
columns = st.columns(2)
for column_index, model_name in enumerate(SUPPORTED_MODELS.keys()):
    with columns[column_index]:
        load_model, preprocess_input, decode_predictions = SUPPORTED_MODELS[
            model_name
        ].values()

        model = load_model()
        image_array = bytes_to_array(image_bytes)
        image_array = prepare_image(image_array, _model_preprocess=preprocess_input)
        prediction = model.predict(image_array)
        prediction_df = pd.DataFrame(decode_predictions(prediction, 5)[0])
        prediction_df.columns = ["label_id", "label", "probability"]
        st.write(f"Predictions for model {model_name}")
        st.dataframe(prediction_df.sort_values(by="probability", ascending=False))
