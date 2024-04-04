SHELL=/bin/bash

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv
	cargo fmt --all && cargo clippy --all-features
	# .venv/bin/python -m ruff check . --fix --exit-non-zero-on-fix
	# .venv/bin/python -m ruff format polars_qt tests
	# .venv/bin/mypy polars_qt tests

test: .venv
	.venv/bin/python -m pytest tests


