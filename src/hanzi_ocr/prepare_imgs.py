from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import os
from hanzi_ocr import utils
import matplotlib.pyplot as plt 

def transform_images_folder(folder_path):

    img_transform = transforms.Compose([
        transforms.Resize(size=(70, 70)),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=folder_path, transform=img_transform)


def plot_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

def view_random_img(img_ord, type, font_name):

    img_transform = transforms.Compose([
        transforms.Resize(size=(70, 70)),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    root = utils.find_project_root()
    path = os.path.join(root, "data/", "hanzi_imgs/", f"{img_ord}_{type}_{font_name}.png")

    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img_transform(img)

        plot_image(img)