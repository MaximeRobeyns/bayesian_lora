.PHONY: install install_all example test test_all mypy lab kernel docs help
.PHONY: publish release pypi

install:  ## Install this package in the current environment
	pip install -e .

install_all:  ## Install everything
	pip install -e ".[all]"

# Examples and tests ===========================================================
# Run: pip install -e ".[examples]"

example:  ## Run through the example script
	python ./examples/example_usage.py

test:  ## Run the unit tests
	@python -m pytest -s tests --typeguard-packages=bayesian_lora -k "not test_example"

test_all:  ## Run all the tests (including the slow ones)
	@python -m pytest -s tests --run-slow

# Development ==================================================================
# Run: pip install -e ".[dev]"

mypy:  ## Run static type checking
	@mypy

lab:  ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=notebooks

kernel:  ## To setup a Jupyter kernel to run notebooks in the project's virtual env
	python -m ipykernel install --user --name bayesian_lora \
		--display-name "bayesian_lora"

pypi:  ## Creates a source distribution and wheel, and uplaods to PyPI
	python3 -m pip install --upgrade build
	python3 -m build
	python3 -m twine upload dist/*

release:  ## Create release for GitHub
	$(eval VERSION := $(shell python -c "import bayesian_lora; print(bayesian_lora.__version__)"))
	git checkout master
	git pull origin master
	git tag -a $(VERSION) -m "Release version $(VERSION)"
	git push origin $(VERSION)

publish: release pypi  ## Publish a new release and PyPI package

# Documentation ================================================================
# Run: pip install -e ".[docs]"

docs:  ## Compile the documentation and start watcher
	@./documentation/writedocs.sh

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
