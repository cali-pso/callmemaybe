import argparse
import os
import json
from src.errors import ParsingError
from src.parser import Parser
from src.orchestrator import Orchestrator


def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup and return argument parser with argparse"""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling: from natural language to JSON."
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
    """
    Execute the primary entry point for batch processing function calls.

    Reads user prompts, utilizes the constrained LLM generator to create valid
    function call mappings, evaluates the success of the translations, writes
    structured results back to a JSON file.
    """
    try:
        arg_parser = setup_arg_parser()
        args = arg_parser.parse_args()

        prompts = Parser.load_prompts(args.input)
        functions = Parser.load_functions(args.functions_definition)
        orchestrator = Orchestrator(
            functions=functions
            )

        final_results = []
        total_prompts = len(prompts)

        print(f"Starting generation for {total_prompts} prompts...\n")

        for i, prompt_obj in enumerate(prompts, start=1):
            print(f"Starting generation with {prompt_obj}")
            orchestrator.generate_with_constraints(prompt_obj)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nWriting results to '{args.output}'...")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)

        print("Done.")

    except ParsingError as e:
        print(f"Error parsing: {e}")
    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
