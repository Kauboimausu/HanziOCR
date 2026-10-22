import json
from argparse import Namespace

import cv2
import numpy as np
import pandas as pd
import pytest

from hanzi_ocr.train_hanzi_model import make_checkpoint_path, train_model

FONTS = ["FontA.ttf", "FontB.ttf"]
CODEPOINTS = [ord("中"), ord("文"), ord("字")]


def make_opts(**overrides):
    """Same names as the script's argparse, any of them can be overridden"""
    opts = {
        "optimizer": "adamw",
        "scheduler": None,
        "metric": "accuracy",
        "lr": 0.001,
        "epochs": 1,
        "patience_epochs": 1,
        "early_stopping_monitor": "valid_metric",
        "batch_size": 3,
        "data_folder": "data",
        "model_name": "test_model",
        "imgs_folder": "imgs",
        "manifest_folder": "manifest",
        "manifest_name": "manifest.csv",
        "holdout_fonts": ["FontA"],
        "random_seed": 0,
    }
    opts.update(overrides)
    return Namespace(**opts)


@pytest.fixture
def project(tmp_path):
    """Fake project: every codepoint rendered (as a blank image) in every font"""
    imgs = tmp_path / "data" / "imgs"
    imgs.mkdir(parents=True)
    rows = []
    for font in FONTS:
        for codepoint in CODEPOINTS:
            file_name = f"{codepoint}_{font}.png"
            cv2.imwrite(str(imgs / file_name), np.full((80, 80), 255, np.uint8))
            rows.append([file_name, codepoint, chr(codepoint), font, "simplified"])

    manifest = tmp_path / "data" / "manifest"
    manifest.mkdir()
    pd.DataFrame(rows, columns=["file name", "codepoint", "text", "font", "type"]).to_csv(
        manifest / "manifest.csv", index=False
    )
    return tmp_path


def test_make_checkpoint_path_creates_folder(tmp_path):
    folder = make_checkpoint_path(tmp_path, "data", "my_model")
    assert (tmp_path / "data" / "model_weights" / "my_model").is_dir()
    assert str(folder) == str(tmp_path / "data" / "model_weights" / "my_model")


def test_make_checkpoint_path_works_if_folder_exists(tmp_path):
    make_checkpoint_path(tmp_path, "data", "my_model")
    make_checkpoint_path(tmp_path, "data", "my_model")


def test_train_model_saves_all_outputs(project):
    train_model(project, make_opts())
    model_folder = project / "data" / "model_weights" / "test_model"
    assert (model_folder / "checkpoint.pt").is_file()
    assert (model_folder / "le.json").is_file()
    assert (model_folder / "train_hyperparameters.json").is_file()
    assert (model_folder / "training_history.json").is_file()
    assert (model_folder / "training_history.png").is_file()


def test_train_model_history_has_one_entry_per_epoch(project):
    train_model(project, make_opts(epochs=2, patience_epochs=5))
    history_path = project / "data" / "model_weights" / "test_model" / "training_history.json"
    history = json.loads(history_path.read_text())
    assert {len(values) for values in history.values()} == {2}


@pytest.mark.parametrize("monitor", ["valid_metric", "train_loss"])
def test_train_model_accepts_both_monitors_with_plateau_scheduler(project, monitor):
    train_model(project, make_opts(early_stopping_monitor=monitor, scheduler="performance"))


def test_train_model_saves_label_encoder_classes(project):
    train_model(project, make_opts())
    le_path = project / "data" / "model_weights" / "test_model" / "le.json"
    assert json.loads(le_path.read_text()) == sorted(CODEPOINTS)


def test_train_model_handles_single_sample_last_batch(project):
    # 3 training images with batch_size 2 leaves a last batch of 1 image
    train_model(project, make_opts(batch_size=2))


def test_train_model_copies_data_generation_parameters(project):
    parameters = {"font_weights": ["Light", "Bold"], "character_size": 100}
    (project / "data" / "manifest" / "generation_parameters.json").write_text(json.dumps(parameters))
    train_model(project, make_opts())
    copied = project / "data" / "model_weights" / "test_model" / "data_generation_parameters.json"
    assert json.loads(copied.read_text()) == parameters


def test_train_model_works_without_data_generation_parameters(project, capsys):
    train_model(project, make_opts())
    assert not (project / "data" / "model_weights" / "test_model" / "data_generation_parameters.json").exists()
    assert "WARNING" in capsys.readouterr().out
