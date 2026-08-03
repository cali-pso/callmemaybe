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
        orchestrator = Orchestrator(functions=functions)

        padded_ids = orchestrator.get_padded_ids(prompts)
        batch_results = orchestrator.generate_with_constraints_batched(
            padded_ids
        )

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nWriting results to '{args.output}'...")

        formatted_results = []
        for prompt_obj, raw_result in zip(prompts, batch_results):
            try:
                parsed_json = json.loads(raw_result)
                formatted_results.append(
                    {
                        "prompt": prompt_obj.prompt,
                        "name": parsed_json.get("name"),
                        "parameters": parsed_json.get("parameters", {}),
                    }
                )
            except json.JSONDecodeError as e:
                print(
                    "JSON validation failed for prompt",
                    f"'{prompt_obj.prompt}': {e}",
                )

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, indent=4)

        print("Done.")

    except ParsingError as e:
        print(f"Error parsing: {e}")
    # except Exception as e:
    #   print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
