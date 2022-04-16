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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dummy = 70

st.set_page_config(
    page_title="Pawpularity of Shelter Animals",
    page_icon="https://play-lh.googleusercontent.com/wUJyDdA2P5rKWUomow__kXlAgx9tCx-i4qcKF-c_rtw_qZNp52AZ223vv9NNBY_ZCky7",
    layout='wide')

sns.set_style("darkgrid")

# @st.cache
def load_image(image_file, format):
    if format == "np":
        return np.array(Image.open(image_file))
    elif format == "tensor":
        return torch.utils.data.DataLoader(image_file)
    elif format == "pil":
        return Image.open(image_file)
    else:
        return image_file


# Write image to disk
def upload_image(img, upload_folder):
    Image.open(img).save(f"./{upload_folder}/{img.name}")


def load_model(filename):
    return torch.load(open(filename, "rb"), map_location=torch.device("cpu"))


def warning_message(blur, occlusion):
    if blur == 1 or occlusion == 1:

        if blur == 1 and occlusion == 1:
            # text = 'Your image appears blury and has some occlusion, you may want to consider a different image'
            text = '<span style="color:Red">Warning: </span> your image appears blury and has some occlusion, you may want to consider a different image'
        elif blur == 1:
            # text = 'Your image appears blurry, you may want to consider a different image'
            text = '<span style="color:Red">Warning: </span> your image appears blurry, you may want to consider a different image'
        else:
            # text = 'There appears to be some occlusion in your image, you may want to consider a different one'
            text = '<span style="color:Red">Warning: </span> there appears to be some occlusion in your image, you may want to consider a different one'

        # st.write("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, 'Warning:') + text)
        st.markdown(text, unsafe_allow_html=True)
    else:
        text = 'There appear to be no problems with your image' # can prob just delete this if no issues
        st.write(text)

def generate_images(df, score, samples):

    df = df.copy()
    
    df = df[(df['Pawpularity'] >= score - 2) & (df['Pawpularity'] <= score + 2)]
    df = df.sample(n=samples*4)
    img_ids = df['Id']
    imgList = []

    idx = 0
    for i in img_ids:
        # st.write(i)
        filepath = './train_images/' + str(i) + '.jpg'
        img = load_image(filepath, 'np')

        w, h, _ = img.shape
        if w != 960 or h != 720:
            continue

        imgList.append(img)
        idx += 1
        if idx == samples:
            break

        
    st.image(imgList, width=250)

    


def main():
    
    # row0_spacer1, row0_1, row0_spacer2 = st.columns(3) #(.15, .5, .005, .75, .005, .5, .005, 1, .15)
    row0_spacer1, row0_1, row0_spacer2 = st.columns((0.25, 0.5, 0.25))
    with row0_1:
        st.title("""Predict Your Pet's Pawpularity!""")
        image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])

    

    if image_file is not None:

        upload_image(image_file, "uploads")

        # Write some metadata about the image - sanity check that it is uploaded
        file_details = {
            "filename": image_file.name,
            "file_type": image_file.type,
            "filesize": image_file.size,
        }

        row1_spacer1, row1_1, row1_2, row1_spacer2 = st.columns((.25, .25, .25, .25))
        with row1_1:
            # st.write(file_details)
            img_np = load_image(image_file, "np")
            st.image(img_np, width=250)
            # st.write(np.shape(np.array(img_np)))
        with row1_2:
            # data
            img = CustomDataset([load_image(image_file, "pil")])
            img = load_image(img, "tensor")
            

            # Models
            # pawpularity
            model = load_model("./eff_net.sav")
            scores = []

             # blur
            model2 = load_model("./eff_net.sav")
            blur_scores = []

            # occlusion
            model3 = load_model("./eff_net.sav")
            occlusion_scores = []

            # scores
            for input in img:
                scores.append(model(input))
                blur_scores.append(model2(input))
                occlusion_scores.append(model3(input))

            pawpularity = float(scores[0][0])
            blur = int(blur_scores[0][0])
            occlusion = int(occlusion_scores[0][0])


            warning_message(blur, occlusion)
            st.write(f"Predicted Pawpularity: {pawpularity:.2f}")


        row2_spacer1, row2_1, row2_spacer2 = st.columns((0.25, 0.5, 0.25))
        with row2_1:
            df = pd.read_csv("./train.csv")

            fig, ax = plt.subplots()
            ax = sns.kdeplot(x=df['Pawpularity'], shade=True)
            ax.axvline(
                x=float(pawpularity),
                color="r",
                linestyle="dashed",
                linewidth=2,
                label="Your Score!",
            )
            ax.set_title("Pawpularity Density")
            ax.legend()
            st.pyplot(fig)


        row3_spacer1, row3_1, row3_spacer2 = st.columns((0.25, 0.5, 0.25))
        with row3_1:
            st.subheader("Similarly Scored Photos")
            generate_images(df, int(pawpularity), 6)


        row3_spacer1, row3_1, row2_spacer3 = st.columns((0.25, 0.5, 0.25))
        with row3_1:
            st.markdown('___')
            about = st.expander('About/Additional Info')
            with about:
                """
                ### What we are trying to do:
                Using modern computer vision techniques, our goal is to create a website that allows users upload
                photos of their pets and determine how popular the pet's post will be, in addition to providing useful
                feedback on how they could improve the photos. Ideally, this tool would help increase the quality of
                pet adoption posts, resulting in more forever homes and less euthanized animals.
                
                ### Why do we do this:
                According to the American Society for the Prevention of Cruelty to Animals (ASPCA), there are
                approximately 920,000 cats and dogs are euthanized in shelters every year. These pets typically have
                online posts made in order to find an appropriate home yet go unnoticed. [PetFinder](https://www.petfinder.com/) is a pet adoption
                website that has recorded the pictures of cats and dogs up for adoption, with each photo containing
                an associated popularity score of the post, which is based upon each pet's page view statistics. The
                problem with calculating each post's popularity this way is that it is purely historical and does not
                provide users with any information.
                """


           


if __name__ == "__main__":
    main()
