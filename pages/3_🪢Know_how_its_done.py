import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import glob
from glob import glob
import seaborn as sns
import matplotlib.cm as cm
import keras
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy import asarray
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import json
from streamlit_lottie import st_lottie
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import base64

from utils import plot_color_hist_features, preprocess_test_image, plot_hog_features

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_img_as_base64("cropped_demo.jpg")

page_bg_img = f"""
<style>

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{bg_img}");
background-position: right; 
background-repeat: norepeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""


st.sidebar.title("**Lisa.ai 🎨**")
st.sidebar.caption("**Paintings. Culture. Heritage. Preserved. ✅**")
st.sidebar.caption("""Made by: \n 
Harsh Ratna \n
 Arya Itkyal \n
Harsh Tripathi""")
st.sidebar.caption("Look behind the scenes of Lisa AI [here](https://blog.streamlit.io/create-a-color-palette-from-any-image/).")
st.sidebar.markdown("---")

st.title("🪢 Let's Dive Deeper!")

st.info("Get a better glimpse to how our Artificial Intelligence model predicts if a painting is real or not.")
st.subheader("How does AI interpret Paintings?")
st.write("""A machine learning model cannot 'see' and understand a painting like a human eye does. \n
AI sees an image in the form of pixels. Pixel is the smallest unit in an image. When we take a digital image of a painting, it is stored as a combination of pixels. Each pixel contains a different number of channels, usually it a mixture of threee colors 'RGB' : Red, green and blue. \n
It takes a painting as an input and stores it as a combination of different pixel values, in a numpy array . """)

st.subheader(""" How does AI extract 'features' from the painting?""")
st.write("A painting has many features like colors, texture, object shape and edges, brush strokes etc.")
st.write("Our AI Model extracts features from the paintings with the help of 3 techniques.")
st.markdown(""" ##### 1. _Color Histogram_
##### 2. _HOG (Histogram of Oriented Gradients)_
##### 3. _CNN (Deep Learning)_
""")
st.header("Let's visualize! 🙈")

gallery_files = glob(os.path.join(".", "defaultimages", "*"))
gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("_", " ") : image_path
    for image_path in gallery_files}

gallery_tab, upload_tab= st.tabs(["Gallery", "Upload"])
with gallery_tab:
    options = list(gallery_dict.keys())
    index_manual = ['ai generated potrait (fake)', 'beautiful landscape by Edvard Munch (real)','inked man (fake)', 'Lady in despair by Edvard Munch (real)','man_in_blue_swirls_(fake)','Potrait of a man AI generated (Real)','The Scream by Edvard Munch (real)']
    file_name = st.selectbox("Select Art", 
                            options=options, index=index_manual.index("man_in_blue_swirls_(fake)"))
    file = gallery_dict[file_name]

    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")

    img = Image.open(file)

with upload_tab:
    file = st.file_uploader("Upload Art", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the file uploader, remove the image URL first.")
#show image
col1, col2 = st.columns(2)
with col1:
    with st.expander("🖼  Artwork", expanded=True):
        st.image(img, use_column_width=True)
        submit_botton = st.button('Submit Image', type = 'primary')
with col2:
    st.subheader("A 3d Plot of Color Features in the painting")
    progress_bar1 = st.progress(0)
if submit_botton:
    image_array = preprocess_test_image(img)
    with col2:
        for perc_completed in range(100):
            time.sleep(0.01)
            progress_bar1.progress(perc_completed+1)
        plot_color_hist_features(image_array)

st.markdown("---")
st.subheader("Plot Showing Features extracted by Histogram of Oriented Gradients (HOG)")
image_array = preprocess_test_image(img)
if submit_botton:
    plot_hog_features(image_array)
st.markdown("---")

st.subheader("Phew! 😮‍💨")
st.write("So this is how our AI model extracts these features from The painting and uses it to predict if it is real or fake.")
