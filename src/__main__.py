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
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID to use (e.g., 'Qwen/Qwen2.5-0.5B').",
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
        orchestrator = Orchestrator(functions=functions, model_name=args.model)

        # 1. Initialize a list to hold all formatted output objects
        final_results = []
        total_prompts = len(prompts)

        print(f"Starting generation for {total_prompts} prompts...\n")

        for i, prompt_obj in enumerate(prompts, start=1):
            print(f"[{i}/{total_prompts}] Translating: '{prompt_obj.prompt}'")

            answer_str = orchestrator.generate_with_constraints(
                prompt_obj.prompt
            )
            print(answer_str)
            try:
                parsed_answer = json.loads(answer_str.strip("\n"))

                result_entry = {
                    "prompt": prompt_obj.prompt,
                    "name": parsed_answer.get("func_name", ""),
                    "parameters": parsed_answer.get("parameters", {}),
                }

                final_results.append(result_entry)
                print("SUCCESS")

            except json.JSONDecodeError:
                print("FAILED (Invalid JSON format)")
            except Exception as e:
                print(f"FAILED ({e})")

        # Ensure output directory exists before saving
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nWriting results to '{args.output}'...")

        # 5. Write the final formatted list to the JSON file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)

        print("Done.")

    except ParsingError as e:
        print(f"Error parsing: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
