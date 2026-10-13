import numpy as np
import pytest
from line_image_generation import take_wiki_excerpt


@pytest.fixture
def rng():
    return np.random.default_rng(0)

def test_excerpt_respects_min_length(rng):
    text = "这是一个很长的测试句子用来检查摘录"
    for _ in range(200):
        excerpt = take_wiki_excerpt(text, 3, 8, rng)
        assert len(excerpt) >= 3


def test_excerpt_respects_max_length(rng):
    text = "这是一个很长的测试句子用来检查摘录"
    for _ in range(200): 
        excerpt = take_wiki_excerpt(text, 3, 8, rng)
        assert len(excerpt) <= 8


def test_excerpt_max_range_is_inclusive(rng):
    text = "这是一个很长的测试句子用来检查摘录"
    lengths = set()
    for _ in range(200):
        excerpt = take_wiki_excerpt(text, 3, 8, rng)
        lengths.add(len(excerpt))
    assert 8 in lengths