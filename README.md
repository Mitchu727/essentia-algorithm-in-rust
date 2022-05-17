# essentia-algorithm-in-rust
Project developed within ZPR course (pol. Zaawansowane programowanie w c ++, eng. Advanced C++ programming) at Warsaw University of Technology

## Environment
To start working you have to install two things:
1. Python poetry - tool for dependency management and packaging in Python. You can install it with:
```bash
   pip install poetry
```
2. Rust - programming language. You can install it with:
```bash
   $ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```
3. make - macOs and Linux already have it, on windows you can do this as described [here](https://www.technewstoday.com/install-and-use-make-in-windows/). 
## Development
To install all needed dependencies via poetry:
```bash
   make install
```
To build project with rust library use:
```bash
   make build
```
To apply changes to rust library:
```bash
   make dev
```
To run tests:
```bash
   make test
```
To generate documentation:
```bash
   make docs
```

To use linters:
```bash
   make lint
```

To format rust code:
```bash
   make fmt
```