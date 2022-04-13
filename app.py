import streamlit as st
from PIL import Image
import os
import numpy as np
import urllib.request
import pickle
from torchvision import transforms as T
import torchvision.models as models
import torch
from dataSet import CustomDataset

@st.cache
def load_image(image_file, format):
    if format == 'np':
        return np.array(Image.open(image_file))
    elif format == 'tensor':
        return torch.utils.data.DataLoader(image_file)
    else:
        return image_file


# Write image to disk
def upload_image(img, upload_folder):
    Image.open(img).save(f'./{upload_folder}/{img.name}')

def load_model(filename):
    return torch.load(open(filename, 'rb'), map_location=torch.device('cpu'))


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
            

            upload_image(image_file, 'uploads')

            # Write some metadata about the image - sanity check that it is uploaded
            file_details = {"filename": image_file.name, "file_type": image_file.type, "filesize": image_file.size}
            st.write(file_details)
            img_np = load_image(image_file, 'np')
            st.image(img_np, width=250)
            st.write(np.shape(np.array(img_np)))

            img = CustomDataset(csv_path='./row.csv', image_folder='./uploads', column_label='Pawpularity')
            img = load_image(img, 'tensor')
            model = load_model('./eff_net.sav')
            for input, label in img:
                score = model(input)
            print(score)
            st.write(f'Predicted Pawpularity: {score}')
            # with open(os.path.join("./tempDir/", image_file.name), "wb") as f:
            #     try:
            #         os.mkdir("./tempDir/", )
            #     except OSError:
            #         f.write((image_file).getbuffer())
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