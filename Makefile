.PHONY: clean-pycache clean

clean: clean-pycache

clean-pycache:
	find . -name __pycache__ -exec rm -f {} +

lint:
	flake8 qpr scripts tests

test:
	pytest

help:
	@echo "clean - Remove artifacts"
	@echo "clean-pycache - Remove Python artifacts"
	@echo "lint - Check style"
	@echo "test - run pytest"
