import json
from typing import List, Any
from pydantic import BaseModel, ValidationError
from src.models import Prompt, FuncDefinition
from src.errors import ParsingError


class Parser(BaseModel):
    """
    Utility Pydantic class for parsing and validating input JSONs.
    """

    @classmethod
    def _read_and_validate_json_list(
        cls, filepath: str, model: Any
    ) -> List[Any]:
        """
        Generic private method to read JSON and ensure it is a list of objects.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ParsingError(f"Expected JSON array in {filepath}.")

            if not data:
                raise ParsingError(f"The JSON array in {filepath} is empty.")

            return [model(**item) for item in data]

        except FileNotFoundError:
            raise ParsingError(f"File {filepath} not found.")
        except json.JSONDecodeError:
            raise ParsingError(
                f"File {filepath} is syntactically incorrect JSON."
            )
        except ValidationError as e:
            raise ParsingError(f"Data validation failed for {filepath}:\n{e}")

    @classmethod
    def load_prompts(cls, filepath: str) -> List[Prompt]:
        """
        Load and validate testing prompts from JSON file.
        """
        return cls._read_and_validate_json_list(filepath, Prompt)

    @classmethod
    def load_functions(cls, filepath: str) -> List[FuncDefinition]:
        """
        Load and validate function definitions from JSON file.
        """
        return cls._read_and_validate_json_list(filepath, FuncDefinition)
