"""Compare the build against the latest release, master and PyPI."""
import json
import urllib.error
import urllib.request

GITHUB = "https://api.github.com/repos/pyscf/gpu4pyscf"
PYPI = "https://pypi.org/pypi/gpu4pyscf-cuda12x/json"
TIMEOUT = 8


def fetch_json(url, timeout=TIMEOUT):
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "auto-gpu4pyscf"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, ValueError, OSError):
        return None


def fetch(fetcher=fetch_json):
    """Fetch the upstream payloads, None for anything unreachable."""
    return {
        "head": fetcher(f"{GITHUB}/commits/master"),
        "release": fetcher(f"{GITHUB}/releases/latest"),
        "pypi": fetcher(PYPI),
    }


def compare(info, payloads, fetcher=fetch_json):
    """Fold the payloads and the recorded build info into one verdict.

    An empty info means nothing is built, or the build predates build-info.json;
    either way nothing can be called newer with any confidence.
    """
    head, release, pypi = payloads.get("head"), payloads.get("release"), payloads.get("pypi")
    if head is None and release is None and pypi is None:
        return None
    verdict = {
        "newer": False,
        "tag": None,
        "released_at": None,
        "pypi_version": None,
        "master": None,
        "master_date": None,
        "master_subject": None,
        "is_current": False,
        "ahead_by": None,
    }
    if release and release.get("tag_name"):
        verdict["tag"] = release["tag_name"]
        verdict["released_at"] = release.get("published_at", "")
    if pypi:
        verdict["pypi_version"] = pypi["info"]["version"]
        installed = info.get("gpu4pyscf")
        if installed and verdict["pypi_version"] != installed:
            verdict["newer"] = True
    if head:
        sha = head["sha"]
        verdict["master"] = sha
        verdict["master_date"] = head["commit"]["committer"]["date"]
        verdict["master_subject"] = head["commit"]["message"].splitlines()[0]
        built = info.get("git_sha", "")
        verdict["is_current"] = bool(built) and sha == built
        if built and not verdict["is_current"]:
            verdict["newer"] = True
            compared = fetcher(f"{GITHUB}/compare/{built}...{sha}")
            if compared and compared.get("ahead_by") is not None:
                verdict["ahead_by"] = compared["ahead_by"]
        elif not built:
            verdict["newer"] = True
    return verdict
