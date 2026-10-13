import shutil
from argparse import Namespace

import pandas as pd
import pytest
from hanzi_image_generation import get_hanzi_list, write_images

from hanzi_ocr.utils import find_project_root

# A real font from the project, it's only read, never modified
FONT_NAME = "ZCOOLKuaiLe-Regular.ttf"


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
