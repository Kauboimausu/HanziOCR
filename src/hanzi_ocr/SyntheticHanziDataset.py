import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
import cv2
from sklearn.preprocessing import LabelEncoder

def plot_image(image, char):
    """Plots the image along with the character it displays"""
    plt.title(char)
    # plt.imshow(image.permute(1, 2, 0))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()
    

class SynthethicHanziDataset(Dataset):

    def __init__(
        self,
        csv_file,
        img_dir,
        split,
        validation_fonts,
        normalization_pipeline,
        augmentation_pipeline=None,
    ):
        """
        Parameters
        -----------
        csv_file: str
            Complete path file to the csv file which contains the whole manifest for the synthetic hanzi dataset generated with hanzi_image_generation.py
        img_dir: str
            Path to the directory in which the images are stored
        split: ["val", "train"]
            The split of the data to be returned
        validation_fonts: list
            List of the names of the fonts that are to be used for validation, and therefore excluded from training
        augmentation_pipeline:
            Stochastic data augmentation pipeline for training
        normalization_pipeline: 
            Data standarization pipeline for the images, resizing, normalization, etc
        """

        hanzi_df = pd.read_csv(csv_file)

        self.split = split
        if split.lower() == "train":
            self.hanzi_df = hanzi_df[hanzi_df["font"] not in validation_fonts]
        elif split.lower() == "valid":
            self.hanzi_df = hanzi_df[hanzi_df["font"] in validation_fonts]
        self.img_dir = img_dir
        self.augmentation_pipeline = augmentation_pipeline
        self.normalization_pipeline = normalization_pipeline
        unique_vals = self.hanzi_df["codepoint"].unique()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.hanzi_df["codepoint"])
        self.validation_fonts = validation_fonts
        

    def __len__(self):
        return len(self.hanzi_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.img_dir, self.hanzi_df["file name"].iloc[index])

        # image = io.imread(img_name) skimage
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # cv2
        char_codepoint = self.hanzi_df["codepoint"].iloc[index]

        if self.split == "train" and self.augmentation_pipeline is not None:
            # image = self.augmentation_pipeline(image) torchvision transform
            image = self.augmentation_pipeline(image=image)["image"]  # albumentations

        image = self.normalization_pipeline(image=image)["image"]
        encoded_class = self.label_encoder.transform([char_codepoint])

        return image, encoded_class[0]
