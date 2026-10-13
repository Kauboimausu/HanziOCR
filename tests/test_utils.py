import pytest

from hanzi_ocr.utils import is_hanzi


@pytest.mark.parametrize("char, expected", [
    ("中", True), 
    ("篇", True),
    ("個", True),
    ("个", True),
    ("A", False), 
    (".", False),
    ("1", False), 
])
def test_is_hanzi(char, expected): 
    assert is_hanzi(char) == expected
