"""The formatters used in every panel."""
from datetime import datetime, timedelta, timezone

import pytest

from auto_gpu4pyscf import ui

NOW = datetime(2026, 8, 27, 18, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (9, "0:09"), (65, "1:05"), (600, "10:00"), (-5, "0:00"), (3661, "61:01")],
)
def test_fmt_dur(seconds, expected):
    assert ui.fmt_dur(seconds) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [(5_000_000_000, "5.0 GB"), (90_000_000, "90 MB"), (0, "0 MB")],
)
def test_human(size, expected):
    assert ui.human(size) == expected


@pytest.mark.parametrize(
    ("delta", "ending"),
    [
        (timedelta(seconds=10), "(just now)"),
        (timedelta(minutes=3), "(3 min ago)"),
        (timedelta(hours=1), "(1 hour ago)"),
        (timedelta(hours=5), "(5 hours ago)"),
        (timedelta(days=4), "(4 days ago)"),
    ],
)
def test_when_relative(delta, ending):
    stamp = (NOW - delta).isoformat()
    assert ui.when(stamp, now=NOW).endswith(ending)


def test_when_handles_missing_and_unparsable():
    assert ui.when("") == "unknown"
    assert ui.when("not a date") == "not a date"


def test_when_accepts_docker_and_git_shapes():
    """docker prints nanoseconds and an offset, git prints seconds and Z."""
    assert ui.when("2026-08-27T17:19:02.123456789+02:00", now=NOW) != "unknown"
    assert ui.when("2026-08-27T07:33:05Z", now=NOW) != "unknown"


def test_colour_helpers_are_reversible():
    if ui.COLOUR:  # only meaningful on a tty
        assert ui.bold("x").endswith("\033[0m")
    else:
        assert ui.bold("x") == "x"
