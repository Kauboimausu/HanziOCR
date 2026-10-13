import argparse

import pytest

import os

from hanzi_ocr.utils import find_project_root, get_fonts_list, is_hanzi, str2bool, delete_images


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