install:
	poetry install
	poetry shell

build: install
	cargo update
	maturin build
	cargo build

dev: install
	maturin develop

test: dev
	cargo test
	poetry run pytest
