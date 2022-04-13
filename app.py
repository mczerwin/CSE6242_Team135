import streamlit as st
from PIL import Image
import os
import numpy as np
import urllib.request
import pickle
from torchvision import transforms as T
import torchvision.models as models
import torch

@st.cache
def load_image(image_file):
    img = np.array(Image.open(image_file))
    return img


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


def main():
    st.title('CSE6242 DVA Project 2022 Spring Team #135')

    st.markdown(
        '''
        <style>
        [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{
            width: 350px
        }
        [data-testid='stSidebar'][aria-expanded='false'] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )
    st.sidebar.title('*customize later: Sidebar Title')
    st.sidebar.subheader('*customize later: Sidebar subheader')
    menu = ["Image", "Dataset", "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image":
        st.subheader("Image")
        # if we want we can customize the sidebard format under choice=="Image", if needed
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 350px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 350px;
                margin-left: -350px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", 'png'])

        if image_file is not None:

            file_details = {"filename": image_file.name, "file_type": image_file.type, "filesize": image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            st.image(img, width=250)
            st.write(np.shape(np.array(img)))

            model = load_model('./eff_net.sav')
            score = model.predict(img)
            st.write(f'Predicted Pawpularity: {score}')
            with open(os.path.join("./tempDir/", image_file.name), "wb") as f:
                try:
                    os.mkdir("./tempDir/", )
                except OSError:
                    f.write((image_file).getbuffer())
        else:
            # optional to show a demo picture here, if needed
            urllib.request.urlretrieve(
                'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/520px-Cat_poster_1.jpg',
                "gfg.png")

            img = Image.open("gfg.png")
            st.image(img, width=250)
            st.write(np.shape(np.array(img)))

        st.sidebar.markdown('---')
        num_inputbox = st.sidebar.number_input('Numeric parameter input box if needed', value=2, min_value=1)
        st.sidebar.markdown('---')
        float_slider = st.sidebar.slider('Float parameter slider if needed', min_value=0.0, max_value=1.0, value=0.5)

    elif choice == "About":
        st.markdown("In this application, we are user ***deep learning*** to...")
        # if we want we can customize the sidebard format under choice=="About", if needed
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 350px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 350px;
                margin-left: -350px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '''
            about this app section, use markdown rich text here\n

            - [Youtube](https://www.youtube.com/watch?v=ObWSZ4rKUE0)
            '''
        )
        st.video('https://www.youtube.com/watch?v=ouzwOd-TbRo')


    elif choice == "Dataset":
        st.markdown("Dataset")

    elif choice == "DocumentFiles":
        st.markdown("DocumentFiles")


if __name__ == '__main__':
    main()