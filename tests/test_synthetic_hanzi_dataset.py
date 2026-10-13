import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

from hanzi_ocr.SyntheticHanziDataset import SynthethicHanziDataset

FONTS = ["FontA.ttf", "FontB.ttf", "FontC.otf"]
CODEPOINTS = [ord("中"), ord("文"), ord("字")]


@pytest.fixture
def fake_data(tmp_path):
    """Writes a tiny manifest and blank images: every codepoint in every font"""
    rows = []
    for font in FONTS:
        for codepoint in CODEPOINTS:
            file_name = f"{codepoint}_{font}.png"
            cv2.imwrite(str(tmp_path / file_name), np.full((80, 80), 255, np.uint8))
            rows.append([file_name, codepoint, chr(codepoint), font, "simplified"])

    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["file name", "codepoint", "text", "font", "type"]).to_csv(
        csv_path, index=False
    )
    le = LabelEncoder().fit(CODEPOINTS)
    return csv_path, tmp_path, le


@pytest.fixture
def normalization_pipeline():
    return A.Compose([A.Resize(64, 64), A.Normalize(mean=0.5, std=0.5), ToTensorV2()])


def make_dataset(fake_data, normalization_pipeline, split, validation_fonts):
    csv_path, img_dir, le = fake_data
    return SynthethicHanziDataset(
        csv_path, img_dir, split, le, validation_fonts, normalization_pipeline
    )


def test_train_split_excludes_validation_fonts(fake_data, normalization_pipeline):
    ds = make_dataset(fake_data, normalization_pipeline, "train", ["FontA.ttf"])
    assert set(ds.hanzi_df["font"]) == {"FontB.ttf", "FontC.otf"}


def test_valid_split_only_has_validation_fonts(fake_data, normalization_pipeline):
    ds = make_dataset(fake_data, normalization_pipeline, "valid", ["FontA.ttf"])
    assert set(ds.hanzi_df["font"]) == {"FontA.ttf"}


def test_splits_do_not_overlap_and_cover_everything(fake_data, normalization_pipeline):
    train = make_dataset(fake_data, normalization_pipeline, "train", ["FontA.ttf"])
    valid = make_dataset(fake_data, normalization_pipeline, "valid", ["FontA.ttf"])
    assert set(train.hanzi_df["file name"]).isdisjoint(valid.hanzi_df["file name"])
    assert len(train) + len(valid) == len(FONTS) * len(CODEPOINTS)


@pytest.mark.parametrize("font_name", ["FontA.ttf", "fonta", "FONTA.TTF", "FontA"])
def test_validation_fonts_ignore_case_and_extension(
    fake_data, normalization_pipeline, font_name
):
    ds = make_dataset(fake_data, normalization_pipeline, "valid", [font_name])
    assert set(ds.hanzi_df["font"]) == {"FontA.ttf"}


def test_split_name_is_case_insensitive(fake_data, normalization_pipeline):
    ds = make_dataset(fake_data, normalization_pipeline, "Train", ["FontA.ttf"])
    assert len(ds) == 6


def test_unknown_validation_font_fails(fake_data, normalization_pipeline):
    with pytest.raises(ValueError):
        make_dataset(fake_data, normalization_pipeline, "valid", ["NotAFont.ttf"])


def test_all_fonts_in_validation_leaves_train_empty(fake_data, normalization_pipeline):
    with pytest.raises(ValueError):
        make_dataset(fake_data, normalization_pipeline, "train", FONTS)


def test_unknown_split_fails(fake_data, normalization_pipeline):
    with pytest.raises(ValueError):
        make_dataset(fake_data, normalization_pipeline, "test", ["FontA.ttf"])


def test_getitem_returns_float_image_and_encoded_label(
    fake_data, normalization_pipeline
):
    ds = make_dataset(fake_data, normalization_pipeline, "train", ["FontA.ttf"])
    image, label = ds[0]
    _, _, le = fake_data
    assert image.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert le.inverse_transform([label])[0] == ds.hanzi_df["codepoint"].iloc[0]
