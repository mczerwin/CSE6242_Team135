import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms as T
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import percentileofscore
import os


st.set_page_config(
    page_title="Pawpularity of Shelter Animals",
    page_icon="https://play-lh.googleusercontent.com/wUJyDdA2P5rKWUomow__kXlAgx9tCx-i4qcKF-c_rtw_qZNp52AZ223vv9NNBY_ZCky7",
    layout="wide",
)

sns.set_style("darkgrid")


def load_image(image_file, format="tensor"):

    if format == "np":
        return np.array(Image.open(image_file))
    else:
        img = Image.open(image_file)
        # img = ImageOps.exif_transpose(img)

        if image_file.type == "image/png":
            img = img.convert("RGB")

        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((64, 64)),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        tensor_image = transform(img).unsqueeze(0)
        return tensor_image


def load_model(filename, device="cpu"):
    return torch.load(filename, map_location=torch.device(device))


def warning_message(blur, occlusion, face):
    if blur == 1 or occlusion == 1 or face == 0:

        if blur == 1:
            text = '<span style="font-size: 20px">❗Your image appears blurry </span>'
            st.markdown(text, unsafe_allow_html=True)

        if occlusion == 1:
            text = '<span style="font-size: 20px">❗️Your pet appears to be partially blocked by undesirable item </span>'
            st.markdown(text, unsafe_allow_html=True)

        if face == 0:
            text = '<span style="font-size: 20px">❗Please confirm a face is visible in the image</span>'
            st.markdown(text, unsafe_allow_html=True)

    else:
        text = '<span style="font-size: 20px; color:#0ABAB5">🎉There appear to be no problems with your image</span>'
        st.markdown(text, unsafe_allow_html=True)


def generate_images(df, score, samples):

    df = df.copy()

    if score >= 35:
        df = df[(df["Pawpularity"] >= score - 3) &
                (df["Pawpularity"] <= score + 3)]
    else:
        df = df[(df["Pawpularity"] >= score - 2) &
                (df["Pawpularity"] <= score + 2)]
    df = df.sample(n=samples * 4)
    img_ids = df["Id"]
    imgList = []

    idx = 0
    for i in img_ids:
        filepath = "./train_images/" + str(i) + ".jpg"
        img = load_image(filepath, "np")

        w, h, _ = img.shape
        if w != 960 or h != 720:
            continue

        imgList.append(img)
        idx += 1
        if idx == samples:
            break

    st.image(imgList, width=280)


def show_value(val, title):
    fig, ax = plt.subplots(figsize=(2, 2))
    wedgeprops = {'width': 0.2, 'edgecolor': 'none', 'linewidth': 3}
    ax.pie([val, 100-val], wedgeprops=wedgeprops,
           startangle=90, colors=['#5DADE2', '#515A5A'])
    plt.text(0, 0, val, ha='center', va='center', fontsize=15)
    plt.title(title, fontsize=10, loc='center')
    st.pyplot(fig)


def main():
    st.title("""Predict Your Pet's Pawpularity!""")
    row0_spacer1, row0_1, row0_spacer2 = st.columns((0.25, 0.5, 0.25))
    with row0_1:
        image_file = st.file_uploader(
            "Upload an image of your pet", type=["jpg", "jpeg", "png"])
    if image_file is not None:

        # Write some metadata about the image - sanity check that it is uploaded
        # file_details = {
        #     "filename": image_file.name,
        #     "file_type": image_file.type,
        #     "filesize": image_file.size,
        # }
        row01_spacer1, row01_1 = st.columns(
            (0.25, 0.75)
        )
        with row01_1:
            st.subheader("Your Pawpularity!")

        row1_spacer1, row1_1, row1_2, row1_spacer2 = st.columns(
            (0.25, 0.2, 0.3, 0.25)
        )

        with row1_1:
            # st.write(file_details)
            img_np = load_image(image_file, "np")
            st.image(img_np, width=200)
            # st.write(np.shape(np.array(img_np)))
        with row1_2:
            # data
            img = load_image(image_file, "tensor")

            # Models
            # pawpularity
            model = load_model("./pawpularity_mse_model")
            model.eval()
            with torch.no_grad():
                scores = model(img) * 100

            # blur
            model_blur = load_model("./blur_detection")
            model_blur.eval()
            with torch.no_grad():
                blur_scores = model_blur(img)

            # occlusion
            model_occ = load_model("./occlusion_detection")
            model_occ.eval()
            with torch.no_grad():
                occlusion_scores = model_occ(img)

            # face detection
            model_face = load_model("./face_detection")
            model_face.eval()
            with torch.no_grad():
                face_scores = model_face(img)

            pawpularity = float(scores)
            blur = int(torch.round(blur_scores))
            occlusion = int(torch.round(occlusion_scores))
            face = int(torch.round(face_scores))

            df = pd.read_csv("./train.csv")
            paw_arr = np.array(df["Pawpularity"])
            paw_per = percentileofscore(paw_arr, pawpularity)

            paw_txt = f""" <p style="font-family:sans-serif><div id="d1"><span style="font-size: 20px"> Your predicted Pawpularity score is </span><span style="font-size:32px;color:#0ABAB5">{pawpularity:.2f}!</span></div> </p>

                """
            st.markdown(paw_txt, unsafe_allow_html=True)

            warning_message(blur, occlusion, face)

        row2_spacer1, row2_1, row2_spacer2 = st.columns((0.25, 0.5, 0.25))
        with row2_1:
            st.subheader(
                "Let's see how this compares to other photos we've seen")
            pct_text = f"""<p style="font-family:sans-serif"> <div id="d2"> <span style="font-size: 20px"> Your pet's photo is better than  </span><span style="font-size: 32px; color:#0ABAB5">{int(paw_per)}% </span><span style="font-size: 20px"> of images we've seen.</span></div></p> """
            st.markdown(pct_text, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax = sns.kdeplot(x=df["Pawpularity"], shade=True)
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

        next_text = f"""<p style="font-family:sans-serif"> <div id="d2"> <span style="font-size: 20px"> Here are some pet photos with similar Pawpularity scores.</span></div></p> """

        row3_spacer1, row3_1, row3_spacer2 = st.columns((0.25, 0.5, 0.25))
        with row3_1:
            st.subheader("Similarly Scored Photos")
            st.markdown(next_text, unsafe_allow_html=True)
            generate_images(df, int(pawpularity), 6)

        st.subheader("Behind the scenes")
        st.write("Here's a look at the data we've processed so far")
        row4_0, row4_1, row4_2, row4_3 = st.columns(
            (0.25, 0.25, 0.25, 0.25))

        with row4_0:
            uploaded_files = len([name for name in os.listdir(
                './uploads') if os.path.isfile(name)])
            photo_count = len(df) + uploaded_files

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.pie([100, 0], wedgeprops={'width': 0.2},
                   startangle=90, colors=['#5DADE2'])
            plt.text(0, 0, photo_count, ha='center', va='center', fontsize=15)
            plt.title("Total Images Processed", fontsize=10, loc='center')
            st.pyplot(fig)

        with row4_1:
            avg_pawpularity = round(df['Pawpularity'].mean(), 2)
            show_value(avg_pawpularity, 'Avg Pawpularity')

        with row4_2:
            pct_blur = len(df[df['Blur'] == 1]) / len(df)
            pct_occlusion = len(df[df['Occlusion'] == 1]) / len(df)
            pct_no_face = len(df[df['Face'] == 0]) / len(df)

            supp_data = {'Issue': ['Blurry', 'Occlusion', 'No Face'], 'Percent of Values': [
                pct_blur, pct_occlusion, pct_no_face]}
            supp_df = pd.DataFrame(supp_data)

            fig, ax = plt.subplots(figsize=(15, 15))
            ax = sns.barplot(
                x='Percent of Values', y='Issue', data=supp_df)
            plt.title("Percent of Values With...")
            st.pyplot(fig)

        row5_spacer1, row5_1, row5_spacer3 = st.columns((0.25, 0.5, 0.25))

        with row5_1:
            st.markdown("___")
            about = st.expander("About/Additional Info")
            with about:
                """
                # What we are trying to do:
                Using modern computer vision techniques, our goal is to create a website that allows users upload
                photos of their pets and determine how popular the pet's post will be, in addition to providing useful
                feedback on how they could improve the photos. Ideally, this tool would help increase the quality of
                pet adoption posts, resulting in more forever homes and less euthanized animals.

                # Why do we do this:
                According to the American Society for the Prevention of Cruelty to Animals(ASPCA), there are
                approximately 920, 000 cats and dogs are euthanized in shelters every year. These pets typically have
                online posts made in order to find an appropriate home yet go unnoticed.
                [PetFinder](https: // www.petfinder.com /) is a pet adoption website that has recorded the pictures of cats
                and dogs up for adoption, with each photo containing an associated popularity score of the post, which
                is based upon each pet's page view statistics. The problem with calculating each post's popularity this
                way is that it is purely historical and does not provide users with any information.
                """


if __name__ == "__main__":
    main()
