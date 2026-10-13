import argparse

import pytest

from hanzi_ocr.utils import find_project_root, get_fonts_list, is_hanzi, str2bool


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
    with pytest.raises(Exception):
        get_fonts_list("data/", "this_folder_does_not_exist/")
