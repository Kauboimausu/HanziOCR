import json
import shutil
from argparse import Namespace

import pandas as pd
import pytest
from hanzi_image_generation import get_font_weights, get_hanzi_list, write_images
from PIL import ImageFont

from hanzi_ocr.utils import GENERATION_PARAMETERS_NAME, find_project_root

# Real fonts from the project, they're only read, never modified
FONT_NAME = "ZCOOLKuaiLe-Regular.ttf"
VARIABLE_FONT_NAME = "NotoSerifSC-VariableFont_wght.ttf"


def load_font(name):
    return ImageFont.truetype(str(find_project_root() / "data" / "fonts" / name), size=40)


def make_opts(**overrides):
    """Same names as the script's argparse, any of them can be overridden"""
    opts = {
        "data_folder": "data",
        "font_location": "fonts",
        "hanzi_location": "characters",
        "hanzi_file_name": "characters.csv",
        "image_save_location": "imgs",
        "manifest_save_location": "manifest",
        "manifest_name": "manifest.csv",
        "error_manifest_name": "error_manifest.csv",
        "character_size": 40,
        "width_padding": 5,
        "height_padding": 5,
        "hanzi_styles": "s",
        "font_weights": ["Light", "Regular", "Bold"],
    }
    opts.update(overrides)
    return Namespace(**opts)


@pytest.fixture
def project(tmp_path):
    """Fake project: one real font, a 'reserved' folder and a tiny character list"""
    fonts = tmp_path / "data" / "fonts"
    fonts.mkdir(parents=True)
    shutil.copy(find_project_root() / "data" / "fonts" / FONT_NAME, fonts / FONT_NAME)
    (fonts / "reserved").mkdir()

    characters = tmp_path / "data" / "characters"
    characters.mkdir()
    pd.DataFrame(
        [[ord("中"), "中", "中"], [ord("爱"), "爱", "愛"], [ord("…"), "…", "…"]],
        columns=["codepoint", "simplified", "traditional"],
    ).to_csv(characters / "characters.csv", index=False)
    return tmp_path


def read_manifest(project, name="manifest.csv"):
    return pd.read_csv(project / "data" / "manifest" / name)


def test_get_hanzi_list_is_indexed_by_codepoint(project):
    df = get_hanzi_list(project, "data", "characters", "characters.csv")
    assert df.loc[ord("中"), "simplified"] == "中"


def test_write_images_renders_every_simplified_hanzi(project):
    write_images(project, make_opts())
    assert set(read_manifest(project)["text"]) == {"中", "爱"}


def test_write_images_saves_the_files_listed_in_manifest(project):
    write_images(project, make_opts())
    for file_name in read_manifest(project)["file name"]:
        assert (project / "data" / "imgs" / file_name).is_file()


def test_write_images_skips_non_hanzi(project):
    write_images(project, make_opts())
    assert "…" not in set(read_manifest(project)["text"])
    assert "…" not in set(read_manifest(project, "error_manifest.csv")["text"])


def test_write_images_skips_reserved_fonts(project):
    write_images(project, make_opts())
    assert set(read_manifest(project)["font"]) == {FONT_NAME}


def test_write_images_logs_missing_glyphs(project):
    # ZCOOL KuaiLe is a simplified-only font, it has no glyph for 愛
    write_images(project, make_opts(hanzi_styles="t"))
    assert "愛" in set(read_manifest(project, "error_manifest.csv")["text"])
    assert "愛" not in set(read_manifest(project)["text"])


def test_write_images_both_styles(project):
    write_images(project, make_opts(hanzi_styles="st"))
    assert set(read_manifest(project)["type"]) == {"simplified", "traditional"}


def test_get_font_weights_single_weight_font_is_one_pass():
    assert get_font_weights(load_font(FONT_NAME), ["Light", "Regular", "Bold"]) == [None]


def test_get_font_weights_variable_font_returns_wanted_names_as_str():
    weights = get_font_weights(load_font(VARIABLE_FONT_NAME), ["Light", "Regular", "Bold"])
    assert weights == ["Light", "Regular", "Bold"]
    assert all(isinstance(weight, str) for weight in weights)


def test_get_font_weights_only_returns_wanted_weights():
    assert get_font_weights(load_font(VARIABLE_FONT_NAME), ["Bold"]) == ["Bold"]


@pytest.fixture
def variable_project(project):
    """Same fake project, plus a variable font"""
    fonts = project / "data" / "fonts"
    shutil.copy(find_project_root() / "data" / "fonts" / VARIABLE_FONT_NAME, fonts / VARIABLE_FONT_NAME)
    return project


def test_write_images_renders_each_weight_of_a_variable_font(variable_project):
    write_images(variable_project, make_opts())
    manifest = read_manifest(variable_project)
    variable_rows = manifest[manifest["font"] == VARIABLE_FONT_NAME]
    assert sorted(variable_rows["weight"].unique()) == ["Bold", "Light", "Regular"]
    # 中 and 爱, once per weight
    assert len(variable_rows) == 2 * 3


def test_write_images_labels_single_weight_fonts_as_default(variable_project):
    write_images(variable_project, make_opts())
    manifest = read_manifest(variable_project)
    assert set(manifest[manifest["font"] == FONT_NAME]["weight"]) == {"default"}


def test_write_images_respects_font_weights_option(variable_project):
    write_images(variable_project, make_opts(font_weights=["Bold"]))
    manifest = read_manifest(variable_project)
    assert set(manifest[manifest["font"] == VARIABLE_FONT_NAME]["weight"]) == {"Bold"}


def test_write_images_file_names_are_unique_across_weights(variable_project):
    write_images(variable_project, make_opts())
    manifest = read_manifest(variable_project)
    assert manifest["file name"].is_unique
    assert len(list((variable_project / "data" / "imgs").iterdir())) == len(manifest)


def test_write_images_saves_generation_parameters(project):
    write_images(project, make_opts(font_weights=["Light", "Bold"]))
    parameters_path = project / "data" / "manifest" / GENERATION_PARAMETERS_NAME
    parameters = json.loads(parameters_path.read_text())
    assert parameters["font_weights"] == ["Light", "Bold"]
    assert parameters["character_size"] == 40
