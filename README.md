*This project has been created as part of the 42 curriculum by rapoggi*

# Call Me Maybe

## Descritption

This project is a first glance at the inner mechanisms used by LLM
related software to provide reliable output even on small models.
It teaches to use a technique called constrained decoding to force
the model into respecting a specific syntax and structure on a
precise type of questions. Here, to generate syntaxically correct
JSON representing a function call needed by the prompt.

## Instructions

To install the necessary packages into a virtual environment managed by uv, run:
```bash
make
```
To run the program with default input/output files found in data, run:
```bash
make run
```
To clean folder of files created by installation, run:
```bash
make clean
```
To execute lint check, run:
```bash
make lint
```

Custom paths for files used by the program can be passed as such:
```bash
uv run python -m src [–functions_definition <function_definition_file>] \
       [–input <input_file>] [–output <output_file>]
```
