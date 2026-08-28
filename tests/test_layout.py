"""The repository layout the scripts and Dockerfile depend on.

A moved file breaks a COPY path or a wrapper without anything failing until a
build is well under way.
"""
import re
import stat

from conftest import ROOT

from auto_gpu4pyscf import paths


def test_repo_root_is_found_from_the_package():
    assert paths.repo_root() == ROOT


def test_expected_directories_exist():
    for relative in ("docker", "scripts", "share", "src/auto_gpu4pyscf", "tests"):
        assert (ROOT / relative).is_dir(), relative


def test_dockerfile_copy_sources_resolve():
    dockerfile = paths.dockerfile().read_text()
    sources = re.findall(r"^COPY (?!--from)(\S+) ", dockerfile, flags=re.MULTILINE)
    assert sources, "no plain COPY lines found; did the Dockerfile change shape?"
    for source in sources:
        assert (ROOT / source).exists(), f"COPY {source} does not resolve from the repo root"


def test_dockerignore_keeps_the_context_small():
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "*" in ignore, "everything should be excluded by default"
    assert "!share/" in ignore, "the image needs share/"
    assert not any(line.strip() == "!env/" for line in ignore)


def test_wrappers_are_executable_and_present():
    expected = ["build.sh", "build_env.sh", "run.sh", "build.ps1", "run.ps1"]
    for name in expected:
        script = paths.scripts() / name
        assert script.is_file(), name
        if name.endswith(".sh"):
            assert script.stat().st_mode & stat.S_IXUSR, f"{name} is not executable"
    assert (ROOT / "menu.sh").stat().st_mode & stat.S_IXUSR


def test_build_scripts_point_at_the_moved_dockerfile():
    for name in ("build.sh", "build.ps1"):
        text = (paths.scripts() / name).read_text()
        assert "-f docker/Dockerfile" in text, name


def test_every_example_has_a_runner_and_is_valid_python():
    """Every example ships a wrapper that runs it without the menu."""
    examples = ROOT / "examples"
    scripts = sorted(examples.glob("*.py"))
    assert scripts, "no examples found"
    for script in scripts:
        runner = script.with_suffix(".sh")
        assert runner.is_file(), f"{script.name} has no runner"
        assert runner.stat().st_mode & stat.S_IXUSR, f"{runner.name} is not executable"
        assert "_run.sh" in runner.read_text(), f"{runner.name} bypasses the shared launcher"
        compile(script.read_text(), script.name, "exec")
    assert (examples / "_run.sh").stat().st_mode & stat.S_IXUSR


def test_examples_keep_their_upstream_licence_header():
    """They are upstream's files, copied unmodified."""
    for script in (ROOT / "examples").glob("*.py"):
        head = script.read_text()[:800]
        assert "Apache License" in head, script.name
        assert "PySCF Developers" in head, script.name


def test_shared_helpers_are_valid_python():
    for name in ("build_info.py", "smoke_test.py"):
        source = (paths.share() / name).read_text()
        compile(source, name, "exec")


def test_state_lives_outside_the_checkout():
    """Generated state must not land in the working tree."""
    for path in (paths.settings_file(), paths.profile_file(), paths.build_log()):
        assert ROOT not in path.parents, f"{path} is inside the checkout"


def test_the_version_is_the_same_everywhere():
    """pyproject, the package and the changelog have to agree, or the shipped
    version has no entry describing it."""
    declared = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(),
                         re.MULTILINE)
    assert declared, "pyproject.toml has no version"
    version = declared.group(1)

    package = re.search(r'^__version__ = "([^"]+)"',
                        (ROOT / "src" / "auto_gpu4pyscf" / "__init__.py").read_text(),
                        re.MULTILINE)
    assert package and package.group(1) == version

    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{version}]" in changelog, f"CHANGELOG.md has no section for {version}"
