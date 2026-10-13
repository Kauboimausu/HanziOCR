from argparse import Namespace

import pandas as pd
import pytest
from make_char_list import make_char_list


def make_opts(**overrides):
    """Same defaults as the script's argparse, any of them can be overridden"""
    opts = {
        "data_folder": "data",
        "hsk_list_folder": "hsk",
        "hsk_level": ["HSK 1.tsv"],
        "save_location": "characters",
        "save_name": "characters.csv",
    }
    opts.update(overrides)
    return Namespace(**opts)


def write_hsk_file(tmp_path, name, rows):
    """Writes an HSK vocabulary file: traditional, simplified, pinyin, definition"""
    folder = tmp_path / "data" / "hsk"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(row) for row in rows]
    (folder / name).write_text("\n".join(lines), encoding="utf-8")


def read_output(tmp_path):
    return pd.read_csv(tmp_path / "data" / "characters" / "characters.csv")


@pytest.fixture
def hsk_1(tmp_path):
    write_hsk_file(tmp_path, "HSK 1.tsv", [
        ["愛好", "爱好", "àihào", "hobby"],
        ["好", "好", "hǎo", "good"],
        ["中文", "中文", "zhōngwén", "Chinese"],
    ])


def test_splits_words_into_unique_characters(tmp_path, hsk_1):
    make_char_list(tmp_path, make_opts())
    assert sorted(read_output(tmp_path)["simplified"]) == sorted("爱好中文")


def test_output_is_sorted_by_codepoint(tmp_path, hsk_1):
    make_char_list(tmp_path, make_opts())
    codepoints = read_output(tmp_path)["codepoint"].tolist()
    assert codepoints == sorted(codepoints)


def test_codepoint_matches_simplified_character(tmp_path, hsk_1):
    make_char_list(tmp_path, make_opts())
    df = read_output(tmp_path)
    assert all(df["codepoint"] == df["simplified"].map(ord))


def test_keeps_traditional_counterpart(tmp_path, hsk_1):
    make_char_list(tmp_path, make_opts())
    df = read_output(tmp_path).set_index("simplified")
    assert df.loc["爱", "traditional"] == "愛"


def test_deduplicates_across_files(tmp_path, hsk_1):
    write_hsk_file(tmp_path, "HSK 2.tsv", [["好人", "好人", "hǎorén", "good person"]])
    make_char_list(tmp_path, make_opts(hsk_level=["HSK 1.tsv", "HSK 2.tsv"]))
    simplified = read_output(tmp_path)["simplified"].tolist()
    assert simplified.count("好") == 1
    assert "人" in simplified


def test_missing_hsk_folder_writes_nothing(tmp_path):
    make_char_list(tmp_path, make_opts())
    assert not (tmp_path / "data" / "characters" / "characters.csv").exists()
