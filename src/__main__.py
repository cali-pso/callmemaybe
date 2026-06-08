import argparse
import os
from src.errors import ParsingError
from src.parser import load_prompts, load_functions


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup and return argument parser with argparse"""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling Tool: natural language into JSON."
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing available functions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the final JSON output will be saved.",
    )
    return parser


def main() -> None:
    try:

        arg_parser = setup_arg_parser()
        args = arg_parser.parse_args()
        prompts = load_prompts(args.input)
        functions = load_functions(args.functions_definition)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    except ParsingError as e:
        print(f"Error parsing: {e}")

if __name__ == "__main__":
    main()
