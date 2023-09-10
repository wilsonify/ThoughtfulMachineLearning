import json
import os.path

import pytest

from s01_wine_scrape.__main__ import happy_path

path_to_here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(name="event")
def event_fixture():
    return json.load(open(f"{path_to_here}/s3-put.json", "r"))


def test_happy_path(event):
    result = happy_path(event)
    assert result == {}
