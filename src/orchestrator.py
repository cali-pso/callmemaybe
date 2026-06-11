import json
from typing import List
from llm_sdk import Small_LLM_Model
from models import FuncDefinition


class Orchestrator:
    def __init__(self, functions: List) -> None:
        self.functions = functions
        self.model = Small_LLM_Model()
        self.vocab = self.load_vocabulary_mapping(
            self.model.get_path_to_vocab_file()
        )
        self.system_prompt = "Meet request by choosing and returning one of the given functions definition"

    @staticmethod
    def load_vocabulary_mapping(vocab_path: str) -> dict[int, str]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)

        id_to_string = {
            int(token_id): token_str
            for token_str, token_id in vocab_json.items()
        }
        return id_to_string

    def generate_with_constraints(self, prompt: str) -> str:
        pass


class PromptMaker:
    def __init__(self) -> None:
        pass

    def get_functions_as_prompt(functions: List[FuncDefinition]) -> str:
        pass


class OutputConstrainer:
    def __init__(self) -> None:
        pass
