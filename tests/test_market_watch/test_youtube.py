import pytest
from market_watch.youtube import get_video_id


@pytest.mark.parametrize(
    "url, expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=invalid_id", None),
        ("invalid_url", None),
    ],
)
def test_get_video_id(url, expected_id):
    assert get_video_id(url) == expected_id
