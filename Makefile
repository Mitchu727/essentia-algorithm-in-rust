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

docs: dev
	cargo doc --open --document-private-items

lint:
	cargo clippy
	poetry run flake8

format:
	cargo fmt
