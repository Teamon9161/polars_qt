SHELL=/bin/bash

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r build.requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv
	cargo fmt --all && cargo clippy --all-features
	ruff check . --fix --exit-non-zero-on-fix
	ruff format polars_qt tests
	# .venv/bin/mypy polars_qt tests

format:
	cargo fmt --all
	cargo clippy --all-features
	ruff check --fix

test: .venv
	pytest tests

debug: 
	maturin develop

release:
	maturin develop --release


