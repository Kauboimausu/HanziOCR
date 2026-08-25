from PIL import Image
import os
from hanzi_ocr import utils
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# def transform_images_folder(folder_path):
# 
#     img_transform = transforms.Compose([
#         transforms.Resize(size=(70, 70)),
#         transforms.CenterCrop(64),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor()
#     ])
# 
#     return ImageFolder(root=folder_path, transform=img_transform)
# 
# def view_random_img(img_ord, type, font_name):
# 
#     img_transform = transforms.Compose([
#         transforms.Resize(size=(70, 70)),
#         transforms.CenterCrop(64),
#         transforms.ToTensor()
#     ])
# 
#     root = utils.find_project_root()
#     path = os.path.join(root, "data/", "hanzi_imgs/", f"{img_ord}_{type}_{font_name}.png")
# 
#     with Image.open(path) as img:
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         img = img_transform(img)
# 
#         plot_image(img)

def plot_image(image, char):
    """
    Plots the image along with the character it displays
    """
    plt.title(char)
    #plt.imshow(image.permute(1, 2, 0))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()
    
class SynthethicHanziDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Parameters
        -----------
        csv_file: str
            Complete path file to the csv file which contains the whole manifest for the synthetic hanzi dataset generated with hanzi_image_generation.py
        img_dir: str
            Path to the directory in which the images are stored
        transform: 
            The transformation pipeline that is to be applied to the dataset
        """
        
        self.hanzi_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.hanzi_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        img_name = os.path.join(self.img_dir, self.hanzi_df["file name"].iloc[index])
        
        #image = io.imread(img_name) skimage 
        image = cv2.imread(img_name, cv2.COLOR_BGR2GRAY) # cv2
        char_codepoint = self.hanzi_df["codepoint"].iloc[index] 
        
        if self.transform is not None:
            # image = self.transform(image) torchvision transform
            image = self.transform(image=image)["image"] # albumentations
        return {"image": image, "char": char_codepoint}



