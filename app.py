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

dummy=70

# @st.cache
def load_image(image_file, format):
    if format == 'np':
        return np.array(Image.open(image_file))
    elif format == 'tensor':
        return torch.utils.data.DataLoader(image_file)
    elif format == 'pil':
        return Image.open(image_file)
    else:
        return image_file


# Write image to disk
def upload_image(img, upload_folder):
    Image.open(img).save(f'./{upload_folder}/{img.name}')

def load_model(filename):
    return torch.load(open(filename, 'rb'), map_location=torch.device('cpu'))


def main():
    st.title('''Predict Your Pet's Pawpularity!''')

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
    st.sidebar.title('CSE6242 DVA Project 2022 Spring Team #135 PetFinder')
    menu = ["Image", "Dataset", "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image":
        st.subheader("Image")
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

            # img = CustomDataset(csv_path='./row.csv', image_folder='./uploads', column_label='Pawpularity')
            img = CustomDataset([load_image(image_file, 'pil')])
            img = load_image(img, 'tensor')
            model = load_model('./eff_net.sav')
            scores = []
            for input in img:
                scores.append(model(input))
         

            for score in scores:
                st.write(f'Predicted Pawpularity: {float(score[0][0]):.2f}')


#TODO possibly add documentation if we keep this about section
    elif choice == "About":
        st.markdown("In this application, we are using ***deep learning*** to...")
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



if __name__ == '__main__':
    main()