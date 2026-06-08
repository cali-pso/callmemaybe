import json
from typing import List
from pydantic import ValidationError
from src.models import Prompt, FuncDefinition
from src.errors import ParsingError


def load_prompts(filepath: str) -> List[Prompt]:
    """
    Load and validate testing prompts from JSON file.

    Args:
        filepath: litteral path to JSON file.

    Returns:
        A list of validated Prompt objects on success, raises error on failure
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            validated_prompts = [Prompt(**item) for item in data]
            return validated_prompts
    except FileNotFoundError:
        raise ParsingError(f"file {filepath} not found.")
    except json.JSONDecodeError:
        raise ParsingError(f"file {filepath} is syntaxically incorrect.")
    except ValidationError as e:
        raise ParsingError(e)


def load_functions(filepath: str) -> List[FuncDefinition]:
    """
    Load and validate function definitions from JSON file.

    Args:
        filepath: litteral path to JSON file.

    Returns:
        A list of FunctionDefinition objects on success, raises error on fail
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            validated_funcs = [FuncDefinition(**item) for item in data]
            return validated_funcs
    except FileNotFoundError:
        raise ParsingError(f"file {filepath} not found.")
    except json.JSONDecodeError:
        raise ParsingError(f"file {filepath} is syntaxically incorrect.")
    except ValidationError as e:
        raise ParsingError(e)
