import os
import argparse
import json

import pandas as pd
import torchmetrics
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from hanzi_ocr.HMNIST import HMNISTModel
from hanzi_ocr.train_nn import train_with_early_stopping
from hanzi_ocr import utils
from hanzi_ocr.SyntheticHanziDataset import SynthethicHanziDataset

import torch.optim as optims
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def make_checkpoint_path(opts):
    """Makes sure the checkpoint path exists and if it doesn't it creates it"""

    root = utils.find_project_root()
    os.makedirs(
        os.path.join(root, "data/", "model_weights/", opts.model_name),
        exist_ok=True,
    )

    return os.path.join(root, "data/", "model_weights/", opts.model_name)

def train_model(opts):
    """Trains the model according to the hyperparameters given"""
    
    root = utils.find_project_root()

    csv_path = os.path.join(
        root, opts.data_folder, opts.manifest_folder, opts.manifest_name
    )

    hanzi_df = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.fit(hanzi_df["codepoint"])
    
    model_save_location = make_checkpoint_path(opts)
    checkpoint_path = os.path.join(model_save_location, "checkpoint.pt")
    json_save_path = os.path.join(model_save_location, "le.json")
    
    with open(json_save_path, "w") as jsonfile:
        json.dump(le.classes_.tolist(), jsonfile, indent=4)
    

    imgs_folder_path = os.path.join(root, opts.data_folder, opts.imgs_folder)
    
    device = get_device()
    num_classes = len(le.classes_)

    model = HMNISTModel(num_classes).to(device)

    match opts.optimizer.lower():
        case "adam":
            optimizer = optims.Adam(
                params=model.parameters(), lr=opts.lr, betas=(0.9, 0.999)
            )
        case "adamw":
            optimizer = optims.AdamW(params=model.parameters(), lr=opts.lr)
        case "nag":
            optimizer = optims.SGD(
                params=model.parameters(), nesterov=True, momentum=0.9, lr=opts.lr
            )
        case "momentum":
            optimizer = optims.SGD(
                params=model.parameters(), nesterov=False, momentum=0.9, lr=opts.lr
            )
        case _:
            raise ValueError(
                "The optimized is not supported, pick one from: 'Adam', 'AdamW', 'NAG', or 'Momentum'"
            )

    match opts.metric.lower():
        case "accuracy":
            metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(device)
        case "f1":
            metric = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes
            ).to(device)
        case "f1 macro":
            metric = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)
        case _:
            raise ValueError("The metric must be specified, pick one from: 'Accuracy', 'F1', or 'F1 Macro'")

    criterion = CrossEntropyLoss()

    augmentation_pipeline = A.Compose(
        [
            A.Illumination(p=0.4),
            A.RandomShadow(p=0.7),
            A.AdditiveNoise(p=0.3),
            A.InvertImg(p=0.15),
            A.MotionBlur(p=0.25),
            A.Defocus(p=0.15),
            A.Perspective(fit_output=True, keep_size=False, p=0.5),
            A.SafeRotate((-30, 30), p=0.6),
        ],
        seed=opts.random_seed,
    )

    normalization_pipeline = A.Compose(
        [
            A.Resize(64, 64),
            A.Normalize(mean=0.5, std= 0.5, max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    train_ds = SynthethicHanziDataset(
        csv_path,
        imgs_folder_path,
        "train",
        le,
        validation_fonts=opts.holdout_fonts,
        normalization_pipeline=normalization_pipeline,
        augmentation_pipeline=augmentation_pipeline,
    )

    valid_ds = SynthethicHanziDataset(
        csv_path,
        imgs_folder_path,
        "valid",
        le,
        opts.holdout_fonts,
        normalization_pipeline=normalization_pipeline,
        augmentation_pipeline=None,
    )

    rng = torch.Generator().manual_seed(opts.random_seed)
    train_loader = DataLoader(train_ds, opts.batch_size, shuffle=True, generator=rng)
    valid_loader = DataLoader(valid_ds, opts.batch_size, shuffle=False)
    
    match opts.scheduler.lower():
        case "performance":
            scheduler = ReduceLROnPlateau(optimizer, mode="max")
        case "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=opts.lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))
        case "none":
            scheduler = None
        case _:
            raise ValueError("The scheduler given is not valid, pick one from 'Performance', 'OneCycle', or 'None'")

    train_with_early_stopping(
        device,
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        epochs=opts.epochs,
        patience=opts.patience_epochs,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="The name of the optimizer that will be used during training, one of 'Adam', 'AdamW', 'NAG', or 'Momentum'",
    )
    
    parser.add_argument(
        "--scheduler",
        type=str,
        default="None",
        help="The type of scheduler that will be used during training, one of 'Performance', 'OneCycle', or 'None'"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="The name of the metric that will be used for validation and early stopping, one of 'Accuracy', 'F1', 'F1 Macro'",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate the optimizer will use for training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="How many epochs the model will be trained for",
    )

    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=10,
        help="How many patience epochs training will have for early stopping",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size that will be used for training",
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/",
        help="Folder in which the data is stored",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being trained, used to know which folder the weights will be saved in",
    )

    parser.add_argument(
        "--imgs_folder",
        type=str,
        default="hanzi_imgs",
        help="Folder in which the synthetic hanzi images are stored",
    )

    parser.add_argument(
        "--manifest_folder",
        type=str,
        default="hanzi_images_manifest",
        help="Folder in which the manifest csv for the images is stored",
    )

    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest.csv",
        help="Name of the manifest file",
    )

    parser.add_argument(
        "--holdout_fonts",
        type=str,
        nargs="+",
        required=True,
        help="The fonts that are to be used for validation during training, insensitive to capitalization, with or without extension",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed used for transformations and training batch shuffling",
    )

    opts = parser.parse_args()
    train_model(opts)


if __name__ == "__main__":
    main()
