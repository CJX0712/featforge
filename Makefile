.PHONY: install test lint demo clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q -W ignore::UserWarning

lint:
	python -m ruff check .

demo:
	python -m featforge.examples.run_demo

clean:
	rm -rf .pytest_cache .ruff_cache benchmark.json
