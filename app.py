# streamlit -> text/image -> backend -> image cap -> combain -> inference -> same preprocessing -> lstm/rnn -> prediction 

import os
import streamlit as st
from PIL import Image

from backend import ToxicityBackend
from database import create_table


st.set_page_config(
    page_title="Cellula Toxicity Classifier",
    page_icon="🛡️",
    layout="centered"
)

# init database
create_table()

# init backend
@st.cache_resource
def load_backend():
    return ToxicityBackend()

backend =load_backend()

# ui
st.title =("Cellula toxicity classifier")
st.write("enter text ,upload an image")

# model selection
model_type = st.selectbox(
    "Choose Model",
    ["LSTM", "RNN"]
)

# text input
text = st.text_area(
    "Enter text:",
    height=150,
    placeholder="Type your text here..."
)

# image input
uploaded_file = st.file_uploader(
    "Upload an image:",
    type=["png", "jpg", "jpeg"]
)

image =None

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

if st.button(
    "Analyze",
    type="primary",
    use_container_width=True
):

    if not text.strip() and image is None:
        st.warning("Please enter text or upload an image")

    else:

        try:
            image_path = None

            if image is not None:
                os.makedirs(
                    "uploads",
                    exist_ok=True
                )

                image_path = os.path.join(
                    "uploads",
                    uploaded_file.name
                )

                image.save(image_path)


            result = backend.classify(
                text=text if text.strip() else None,
                image=image,
                image_path=image_path,
                model_type=model_type
            )


            st.subheader("Prediction")
            st.success(result["label"])
            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}"
            )

            st.write(f"Model: **{result['model_type']}**")
            st.write(f"Input type: **{result['input_type']}**")

            if result["image_caption"]:

                st.subheader("Image Caption")
                st.info(result["image_caption"])

            with st.expander("View text sent to the model"):
                st.write(result["combined_text"])

        except NotImplementedError as e:
            st.warning(str(e))

        except Exception as e:
            st.error(f"An error occurred: {e}")