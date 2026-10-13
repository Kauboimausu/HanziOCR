import opencc
import pytest
from line_image_generation import (
    clean_entry_text,
    get_excerpt_hanzi_proportion,
    take_wiki_excerpt,
)

TEXT = "这是一个很长的测试句子用来检查摘录"


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
