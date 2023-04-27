import streamlit as st
from streamlit_lottie import st_lottie
import json
from PIL import Image
import base64

st.set_page_config(
    page_title = "AI Generated Forged Painting Detection App",
    page_icon = "🎨"
)


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

st.markdown(page_bg_img, unsafe_allow_html = True)





def load_lottiefile(filepath : str):
    with open(filepath, "r") as f:
        return json.load(f)

st.image('cover page for project.png')
st.sidebar.markdown("Select a page above.")

st.sidebar.title("**Lisa.ai 🎨**")
st.sidebar.caption("**Paintings. Culture. Heritage. Preserved. ✅**")
st.sidebar.caption("""Made by: \n 
Harsh Ratna \n
 Arya Itkyal \n
Harsh Tripathi""")
st.sidebar.caption("Look behind the code of Lisa AI [here](https://blog.streamlit.io/create-a-color-palette-from-any-image/).")
col1, col2= st.columns(2)

with col1:
   st.header("AI-Generated Forged Painting Detection")
   st.markdown("**Paintings. Culture. Heritage. Preserved. ✅**")
   st.markdown('''_Paintings have been an integral part of human culture for centuries. They have been used to tell stories, capture emotions, and depict historical events._ \n_The 
problem of AI- Generated fake paintings is a serious issue that affects both the art market and individual buyers._ \n
_Forgeries can range from low-cost copies to expensive replicas that fetch millions of dollars at auction.''')

with col2:
   lottie_1 = load_lottiefile('lottiefiles/97353-colors-fork.json')
   st.markdown("###")
   st_lottie(lottie_1)

st.info('Use our tool to detect if your Million-dollar painting might be fake or not', icon = "ℹ️")
st.markdown('---')