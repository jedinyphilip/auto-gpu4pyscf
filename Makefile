# Development entry points. The user-facing ones are ./menu.sh and scripts/.
#
# `make install` builds a project virtualenv rather than installing into the
# system python, which recent distributions refuse (PEP 668). Every other
# target uses that venv when it exists and falls back to $(PYTHON) otherwise,
# so `make test` works on a bare checkout too.
PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
PY := $(shell test -x $(VENV_PYTHON) && echo $(VENV_PYTHON) || echo $(PYTHON))
SHELLCHECK := $(shell test -x $(VENV)/bin/shellcheck && echo $(VENV)/bin/shellcheck || echo shellcheck)

.PHONY: help install lint format test check build env menu clean

help:
	@echo "install   create .venv and install the package with dev extras"
	@echo "lint      ruff check + shellcheck, the same checks CI runs"
	@echo "format    ruff format"
	@echo "test      pytest"
	@echo "check     lint + test + dockerfile check"
	@echo "build     build the docker image for this machine"
	@echo "env       build the native environment for this machine"
	@echo "menu      run the menu from the checkout"
	@echo "clean     remove caches and build artefacts (never the image or env/)"

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

install: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --quiet --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	@echo "installed: $(VENV)/bin/auto-gpu4pyscf"

lint:
	$(PY) -m ruff check .
	$(SHELLCHECK) menu.sh scripts/*.sh examples/*.sh

format:
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

check: lint test
	docker build --check --build-arg CUDA_ARCH=120-real -f docker/Dockerfile .

build:
	./scripts/build.sh

env:
	./scripts/build_env.sh

menu:
	./menu.sh

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -name __pycache__ -type d -not -path "./env/*" -exec rm -rf {} +
