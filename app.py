import streamlit as st
from PIL import Image
import os
import numpy as np


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("File Upload Tutorial")

    menu = ["Image","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Images", tyipe=["jpeg"])

        if image_file is not None:
            file_details = {"filename":image_file.name, "file_type":image_file.type, "filesize":image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            st.image(img,width=250)
            st.write(np.shape(np.array(img)))

            with open(os.path.join("images", image_file.name), "wb") as f:
                f.write((image_file).getbuffer())

    elif choice == "Dataset":
        st.subheader("Dataset")
    
    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")


if __name__ == '__main__':
    main()