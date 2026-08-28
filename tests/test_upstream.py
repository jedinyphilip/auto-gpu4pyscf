"""Compare what is built against what upstream has."""
from auto_gpu4pyscf import upstream

BUILT_SHA = "551d9bb165941a49c0b3d1281a3705680487b1b3"
NEWER_SHA = "aaaa111122223333444455556666777788889999"


def payloads(sha=BUILT_SHA, pypi_version="1.8.1", tag="v1.8.1"):
    return {
        "head": {
            "sha": sha,
            "commit": {
                "committer": {"date": "2026-08-27T07:33:05Z"},
                "message": "fix invalid load (#876)\n\nbody",
            },
        },
        "release": {"tag_name": tag, "published_at": "2026-08-10T04:19:00Z"},
        "pypi": {"info": {"version": pypi_version}},
    }


def info(version="1.8.1", sha=BUILT_SHA):
    return {"gpu4pyscf": version, "git_sha": sha}


def test_up_to_date():
    verdict = upstream.compare(info(), payloads())
    assert verdict["is_current"] is True
    assert verdict["newer"] is False
    assert verdict["tag"] == "v1.8.1"
    assert verdict["master_subject"] == "fix invalid load (#876)"


def test_behind_master_counts_commits():
    def fetcher(url):
        assert "compare" in url
        return {"ahead_by": 7}

    verdict = upstream.compare(info(), payloads(sha=NEWER_SHA), fetcher=fetcher)
    assert verdict["newer"] is True
    assert verdict["is_current"] is False
    assert verdict["ahead_by"] == 7


def test_newer_release_on_pypi():
    verdict = upstream.compare(info(version="1.7.0"), payloads())
    assert verdict["newer"] is True
    assert verdict["pypi_version"] == "1.8.1"


def test_unknown_installed_version_is_not_called_newer():
    """An image predating build-info.json records no version to compare."""
    verdict = upstream.compare({}, payloads())
    assert verdict["pypi_version"] == "1.8.1"
    # A rebuild is offered because nothing is built, not from a version diff.
    assert verdict["is_current"] is False


def test_everything_unreachable():
    assert upstream.compare(info(), {"head": None, "release": None, "pypi": None}) is None


def test_partial_outage_still_reports_what_it_has():
    only_github = payloads()
    only_github["pypi"] = None
    verdict = upstream.compare(info(), only_github)
    assert verdict["pypi_version"] is None
    assert verdict["master"] == BUILT_SHA


def test_fetch_uses_one_call_per_source():
    calls = []

    def fetcher(url):
        calls.append(url)
        return {}

    upstream.fetch(fetcher=fetcher)
    assert len(calls) == 3
    assert any("commits/master" in url for url in calls)
    assert any("releases/latest" in url for url in calls)
    assert any("pypi.org" in url for url in calls)
