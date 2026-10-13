import shutil
from argparse import Namespace

import opencc
import pandas as pd
import pytest
from datasets import Dataset
from line_image_generation import (
    clean_entry_text,
    get_excerpt_hanzi_proportion,
    load_wikipedia_data,
    take_wiki_excerpt,
    write_images,
)

from hanzi_ocr.utils import find_project_root

TEXT = "这是一个很长的测试句子用来检查摘录"
FONT_NAME = "ZCOOLKuaiLe-Regular.ttf"
WIKI_PAGES = {
    "pageid": [1, 2, 3],
    "markdown": [
        "中文中文中文中文中文\n\n中国人中国人中国人",
        "this page is only latin text",
        "大家好大家好大家好大家好",
    ],
}


def test_excerpt_respects_min_length(rng):
    for _ in range(200):
        excerpt = take_wiki_excerpt(TEXT, 3, 8, rng)
        assert len(excerpt) >= 3


def test_excerpt_respects_max_length(rng):
    for _ in range(200):
        excerpt = take_wiki_excerpt(TEXT, 3, 8, rng)
        assert len(excerpt) <= 8


def test_excerpt_max_range_is_inclusive(rng):
    lengths = set()
    for _ in range(200):
        excerpt = take_wiki_excerpt(TEXT, 3, 8, rng)
        lengths.add(len(excerpt))
    assert 8 in lengths


def test_excerpt_is_a_substring_of_the_entry(rng):
    for _ in range(200):
        assert take_wiki_excerpt(TEXT, 3, 8, rng) in TEXT


def test_excerpt_can_reach_the_end_of_the_entry(rng):
    endings = set()
    for _ in range(500):
        endings.add(take_wiki_excerpt(TEXT, 3, 3, rng)[-1])
    assert TEXT[-1] in endings


def test_excerpt_min_and_max_can_be_equal(rng):
    assert len(take_wiki_excerpt(TEXT, 5, 5, rng)) == 5


def test_excerpt_returns_none_when_entry_is_too_short(rng):
    assert take_wiki_excerpt("中文", 3, 8, rng) is None


def test_excerpt_returns_none_when_entry_is_none(rng):
    assert take_wiki_excerpt(None, 3, 8, rng) is None


def test_excerpt_returns_whole_entry_when_length_equals_min(rng):
    assert take_wiki_excerpt("中文字", 3, 8, rng) == "中文字"


@pytest.mark.parametrize("excerpt, expected", [
    ("中文字", 100),
    ("abc", 0),
    ("中a", 50),
    ("中文, ok", 100 * 2 / 6),
])
def test_excerpt_hanzi_proportion(excerpt, expected):
    assert get_excerpt_hanzi_proportion(excerpt) == pytest.approx(expected)


@pytest.fixture(scope="module")
def converters():
    return opencc.OpenCC("t2s.json"), opencc.OpenCC("s2t.json")


def test_clean_entry_removes_markdown(converters):
    [(splits, _)] = clean_entry_text("**中文**__字__##", "s", *converters)
    assert splits == ["中文字"]


def test_clean_entry_removes_latex_only(converters):
    [(splits, _)] = clean_entry_text("$x$中文$y$", "s", *converters)
    assert splits == ["中文"]


def test_clean_entry_splits_by_newline(converters):
    [(splits, _)] = clean_entry_text("中文\n汉字", "s", *converters)
    assert splits == ["中文", "汉字"]


def test_clean_entry_simplified_is_simplified(converters):
    [(splits, style)] = clean_entry_text("這是測試", "s", *converters)
    assert style == "simplified"
    assert splits == ["这是测试"]


def test_clean_entry_traditional_is_traditional(converters):
    [(splits, style)] = clean_entry_text("这是测试", "t", *converters)
    assert style == "traditional"
    assert splits == ["這是測試"]


def test_clean_entry_both_styles_in_order(converters):
    result = clean_entry_text("这是测试", "st", *converters)
    assert result == [(["这是测试"], "simplified"), (["這是測試"], "traditional")]


def make_opts(**overrides):
    """Same names as the script's argparse, any of them can be overridden"""
    opts = {
        "data_folder": "data",
        "font_location": "fonts",
        "zh_wiki_location": "wiki",
        "img_save_location": "imgs",
        "manifest_save_location": "manifest",
        "manifest_name": "manifest.csv",
        "error_manifest_name": "error_manifest.csv",
        "pages_per_font": 3,
        "character_size": 30,
        "width_padding": 5,
        "height_padding": 5,
        "snippet_length_range": [3, 5],
        "min_hanzi_proportion": 70,
        "hanzi_styles": "s",
        "random_seed": 0,
    }
    opts.update(overrides)
    return Namespace(**opts)


@pytest.fixture
def project(tmp_path):
    """Fake project: one real font and a tiny Wikipedia dataset saved to disk"""
    fonts = tmp_path / "data" / "fonts"
    fonts.mkdir(parents=True)
    shutil.copy(find_project_root() / "data" / "fonts" / FONT_NAME, fonts / FONT_NAME)
    Dataset.from_dict(WIKI_PAGES).save_to_disk(str(tmp_path / "data" / "wiki"))
    return tmp_path


def read_manifest(project):
    return pd.read_csv(project / "data" / "manifest" / "manifest.csv")


def test_load_wikipedia_data_returns_saved_pages(project):
    ds = load_wikipedia_data(project, "data", "wiki")
    assert ds["pageid"] == WIKI_PAGES["pageid"]


def test_write_images_excerpts_come_from_the_pages(project):
    write_images(project, make_opts())
    texts = read_manifest(project)["text"]
    assert len(texts) > 0
    assert all(any(text in page for page in WIKI_PAGES["markdown"]) for text in texts)


def test_write_images_respects_snippet_length_range(project):
    write_images(project, make_opts())
    assert read_manifest(project)["text"].str.len().between(3, 5).all()


def test_write_images_filters_low_hanzi_proportion(project):
    write_images(project, make_opts())
    proportions = read_manifest(project)["text"].map(get_excerpt_hanzi_proportion)
    assert (proportions >= 70).all()


def test_write_images_saves_the_files_listed_in_manifest(project):
    write_images(project, make_opts())
    for file_name in read_manifest(project)["file name"]:
        assert (project / "data" / "imgs" / file_name).is_file()
