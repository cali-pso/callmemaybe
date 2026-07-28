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

## Resources

<https://huggingface.co/>  
<https://geeks4geeks.com/>  
For general knowledge about the principles of LLM generation.

## Algorithm Explanation

In my project I chose to alternate between proper constrained decoding
(manipulating logits obtained from the model) and "skipping" (safely predicting
whole chunks of json and artificially appending them to generated tokens) to
gain speed by reducing drastically the amount of tokens needed to generate a whole
function call.
Constrained decoding takes place when the "cursor" (current position of the model
in the text) is filling for a parameter. Depending on the type of said parameter,
my OutputConstrainer class restricts possible tokens to, for integers as an exemple,
the numbers 0 through 9 and the comma - for ending the parameter's generation - but
excluding the point, to prevent accidental generation of a float. Same goes for floats
and strings.
The skipping part predicts parts like "name", "parameters", the name of the parameters and
json-syntax specific characters to relieve the model of the answer's longest-to-generate parts.
The model handles solely the choice of the function and the filling of it's parameters.

## Design Decisions

The most important design decision I've had to make in this project is the implementation
of a "skipping" phase as discussed above. This is what made my solution solve 11 prompts in
under 5 minutes as requested by the subject.
OOP has been used throughought the project as it is good practice and allows for an easy-to-read
organisation of code bits.

## Performance Analysis

Considering 42 Perpignan's cluster's computers as standard hardware:
My solution resolves the given 11 prompts in ~3min30 and the moulinette's "private" prompts in
under 4min30. This has been achievable thanks to the "skipping" technique.
As for accuracy and reliability, it chooses perfectly both the function and it's arguments
in far over 90% of the time.
In both time and accuracy, this implementation meets the subject's requirements.

## Challenges Faced

The main challenge has been the generation time. In my first tests, I was getting over 10 minutes
for a single prompt. Through several layers of optimisation and design choices oriented for
performance and efficiency, I managed to get this number down to less than 30 seconds per prompt
on the slowest hardware. I have not faced major challenges regarding the accuracy of answers :
in fact the model did very well without major constrained decoding or skipping.

## Testing Strategy

Testing has been lead in a "bruteforce" way as I just sent as many and diverse prompts as possible,
validating results by looking at them as closely as possible. I also ran and checked the moulinette's
prompt and am now in a position to say that my implementation succeeds in 99% of cases.

## Exemple Usage

For hassle free usage, replace files in data/input/ by your appropriate custom files and run
```bash
make run
```
or
```bash
uv sync
uv run python3 -m src
```

If you need the headache of a long command, run:
```bash
uv run python3 -m src --functions_definition FUNCTIONS_DEF --input INPUT --output OUTPUT
```
Each of these arguments are optional and will default to data/input/*
and data/output
