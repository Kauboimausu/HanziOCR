import os
import argparse
import pandas as pd 
import torch.optim as optims
from hanzi_ocr.HMNIST import HMNISTModel
from hanzi_ocr.train_nn import train_with_early_stopping, evaluate
import torchmetrics
from torch.nn import CrossEntropyLoss
import torch
from src.hanzi_ocr import utils
from torchvision import transforms
import albumentations as A 
from albumentations.pytorch import ToTensorV2

def make_transform(rng_seed):
    pytorch_train_transform = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    seed=rng_seed,
    )
    return pytorch_train_transform

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backend.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def obtain_number_of_classes(opts):
    """Loads the csv and counts the number of classes the model has to have"""
    root = utils.find_project_root()
    hanzi_df = pd.read_csv(os.path.join(root, opts.data_folder, opts.manifest_folder, opts.manifest_name))
    unique_codepoints = hanzi_df["codepoint"]
    return len(set(list(unique_codepoints)))

def train_model(opts):
    """Trains the model according to the hyperparameters given"""
    device = get_device()
    num_classes = obtain_number_of_classes(opts)
    
    model = HMNISTModel(num_classes).to(device)
    
    match optimizer.lower():
        case "adam":
            optimizer = optims.Adam(params=model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
        case "adamw":
            optimizer = optims.AdamW(params=model.parameters())
        case "nag":
            optimizer = optims.SGD(params=model.parameters(), nesterov=True, momentum=0.9, lr=opts.lr)
        case "momentum":
            optimizer = optims.SGD(params=model.parameters(), nesterov=False, momentum=0.9, lr=opts.lr)
    
    match metric.lower():
        case "accuracy":
            metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        case "f1":
            metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(device)
        case "f1_macro":
            metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="micro").to(device)
    
    criterion = CrossEntropyLoss()
    train_with_early_stopping(device, model, 1, 1, criterion, metric, optimizer, None, opts.epochs, opts.patience)
    
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="The name of the optimizer  that will be used during training"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="Accuracy",
        help="The name of the metric that will be used for validation and early stopping"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="The learning rate the optimizer will use for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="How many epochs the model will be trained for"
    )
    
    parser.add_argument(
        "patience_epochs",
        type=int,
        default=10,
        help="How many patience epochs training will have for early stopping"
    )
    
    parser.add_argument(
        "batch_size",
        type=int,
        default=128,
        help="Batch size that will be used for training"
    )
    
    parser.add_argument(
        "--data_folder",
        type=str, 
        default="data/",
        help="Folder in which the data is stored"
    )
    
    parser.add_argument(
        "--model_weights_folder",
        type=str,
        default="hanzi_model_weights/",
        help="Folder in which the model's weight will be stored"
    )
    
    parser.add_argument(
        "imgs_folder",
        type=str,
        default="hanzi_imgs",
        help="Folder in which the synthtic hanzi images are stored"
    )
    
    parser.add_argument(
        "manifest_folder",
        type=str,
        default="hanzi_images_manifest",
        help="Folder in which the manifest csv for the images is stored"
    )
    
    parser.add_argument(
        "manifest_name",
        type=str,
        default="manifest.csv",
        help="Name of the manifest file"
    )
    
    opts = parser.parse_args()
    train_model(opts)
    
    
if __name__ == "__main__":
    main()