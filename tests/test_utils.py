import argparse

import numpy as np
import pytest
from PIL import ImageFont

from hanzi_ocr.utils import (
    delete_images,
    find_project_root,
    get_fonts_list,
    is_hanzi,
    render_text,
    str2bool,
)


@pytest.mark.parametrize("char, expected", [
    ("中", True),
    ("篇", True),
    ("個", True),
    ("个", True),
    ("A", False),
    (".", False),
    ("1", False),
    ("…", False),
    ("。", False),
    ("﹁", False),
])
def test_is_hanzi(char, expected):
    assert is_hanzi(char) == expected


@pytest.mark.parametrize("value", ["yes", "True", "t", "Y", "1", True])
def test_str2bool_accepts_true_values(value):
    assert str2bool(value) is True


@pytest.mark.parametrize("value", ["no", "False", "f", "N", "0", False])
def test_str2bool_accepts_false_values(value):
    assert str2bool(value) is False


def test_str2bool_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_find_project_root_contains_pyproject():
    assert (find_project_root() / "pyproject.toml").exists()


def test_get_fonts_list_fails_on_missing_folder():
    with pytest.raises(FileNotFoundError):
        get_fonts_list(find_project_root(), "data/", "this_folder_does_not_exist/")


def test_delete_images_empties_folders(tmp_path):
    imgs = tmp_path / "data" / "imgs"
    
    imgs.mkdir(parents=True)
    
    (imgs / "a.png").touch()
    (imgs / "A.png").write_text("hello world")
    
    manifest = tmp_path / "data" / "manifest"
    manifest.mkdir(parents=True)
    
    (manifest / "manifest1.csv").write_text("a, b, c")
    (manifest / "manifestA.csv").touch()
    
    delete_images(tmp_path, "data", "imgs", "manifest")
    
    assert list(imgs.iterdir()) == []
    assert list(manifest.iterdir()) == []


def test_delete_images_keeps_parent_folder(tmp_path):
    imgs = tmp_path / "data" / "imgs"

    imgs.mkdir(parents=True)

    (imgs / "a.png").touch()
    (imgs / "A.png").write_text("hello world")

    manifest = tmp_path / "data" / "manifest"
    manifest.mkdir(parents=True)

    (manifest / "manifest1.csv").write_text("a, b, c")
    (manifest / "manifestA.csv").touch()

    delete_images(tmp_path, "data", "imgs", "manifest")

    assert imgs.exists()
    assert manifest.exists()


def test_delete_images_deletes_subfolders(tmp_path):
    imgs = tmp_path / "data" / "imgs"

    imgs.mkdir(parents=True)

    imgs_subfolder = imgs / "subfolder1"
    imgs_subfolder.mkdir(parents=True)

    manifest = tmp_path / "data" / "manifest"
    manifest.mkdir(parents=True)

    manifest_subfolder = manifest / "subfolderA"
    manifest_subfolder.mkdir(parents=True)


    delete_images(tmp_path, "data", "imgs", "manifest")

    assert list(imgs.iterdir()) == []
    assert list(manifest.iterdir()) == []


def test_delete_images_creates_missing_folders(tmp_path):
    delete_images(tmp_path, "data", "imgs", "manifest")

    assert (tmp_path / "data" / "imgs").is_dir()
    assert (tmp_path / "data" / "manifest").is_dir()


def test_delete_images_does_not_touch_other_folders(tmp_path):
    data = tmp_path / "data"
    (data / "imgs").mkdir(parents=True)
    (data / "imgs" / "a.png").touch()
    (data / "fonts").mkdir()
    (data / "fonts" / "font.ttf").write_text("keep me")
    (data / "readme.txt").write_text("keep me too")

    delete_images(tmp_path, "data", "imgs", "manifest")

    assert (data / "fonts" / "font.ttf").read_text() == "keep me"
    assert (data / "readme.txt").read_text() == "keep me too"


def test_get_fonts_list_lists_folder_contents(tmp_path):
    fonts = tmp_path / "data" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "A.ttf").touch()
    (fonts / "B.otf").touch()

    assert sorted(get_fonts_list(tmp_path, "data", "fonts")) == ["A.ttf", "B.otf"]

@pytest.fixture(scope="module")
def font():
    """A real font from the project, it's only read, never modified"""
    font_path = find_project_root() / "data" / "fonts" / "ZCOOLKuaiLe-Regular.ttf"
    return ImageFont.truetype(str(font_path), size=100)


def test_render_text_returns_rgb_image(font):
    assert render_text("中", font, 50, 50).mode == "RGB"


def test_render_text_single_character_is_roughly_square(font):
    width, height = render_text("中", font, 50, 50).size
    assert 0.75 < width / height < 1.33


def test_render_text_line_is_wider_than_tall(font):
    width, height = render_text("这是一个测试", font, 50, 50).size
    assert width > 3 * height


def test_render_text_corners_are_white(font):
    pixels = np.array(render_text("中", font, 50, 50))
    for corner in (pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]):
        assert corner.tolist() == [255, 255, 255]


def test_render_text_draws_black_pixels(font):
    pixels = np.array(render_text("中", font, 50, 50))
    assert (pixels == 0).all(axis=-1).any()


def test_render_text_text_stays_inside_the_padding(font):
    pixels = np.array(render_text("中", font, 50, 50))
    padding = 40  # a bit less than the real padding, to allow for antialiasing
    assert (pixels[:padding] == 255).all()
    assert (pixels[-padding:] == 255).all()
    assert (pixels[:, :padding] == 255).all()
    assert (pixels[:, -padding:] == 255).all()


def test_render_text_width_grows_with_width_padding(font):
    narrow = render_text("中", font, 50, 50)
    wide = render_text("中", font, 60, 50)
    assert wide.width - narrow.width == 20
    assert wide.height == narrow.height


def test_render_text_height_grows_with_height_padding(font):
    short = render_text("中", font, 50, 50)
    tall = render_text("中", font, 50, 60)
    assert tall.height - short.height == 20
    assert tall.width == short.width
