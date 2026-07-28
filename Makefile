PYTHON = uv run python3
FLAKE = uv run flake8
MYPY = uv run mypy
SRC = src

export HF_HUB_OFFLINE=1
export OFFLINE_MODE=True

install:
	uv -n sync
	uv run hf download Qwen/Qwen3-0.6B

run:
	HF_HUB_OFFLINE=1
	$(PYTHON) -m $(SRC)

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	- $(FLAKE) $(SRC)
	- $(MYPY) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC)

lint-strict:
	- $(FLAKE) $(SRC)
	- $(MYPY) --strict $(SRC)
