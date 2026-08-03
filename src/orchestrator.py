import json
import os
from typing import List, Any
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.generator import Generator
from src.output_constrainer import OutputConstrainer
from src.prompt_maker import PromptMaker
from src.models import Prompt


class Orchestrator:
    def __init__(self, functions: List) -> None:
        self.functions = functions
        self.model = Small_LLM_Model()
        self.model_path = self.get_model_path(self.model)
        self.vocab = self.load_vocabulary_mapping(
            self.model_path + "vocab.json"
        )
        self.tokenizer = self.load_tokenizer(
            self.model_path + "tokenizer.json"
        )
        self.constrainer = OutputConstrainer(
            vocab=self.vocab, model=self.model, functions=self.functions
        )
        self.generator = Generator(
            self.model, self.tokenizer, self.constrainer
        )

    @staticmethod
    def get_model_path(model: Small_LLM_Model) -> Any:
        tokenizer_path = model.get_path_to_tokenizer_file()
        return os.path.dirname(tokenizer_path) + "/"

    def load_tokenizer(self, tokenizer_path: str) -> Any:
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)
        self.eos_token = tokenizer["added_tokens"][0]["id"]

        return tokenizer

    @staticmethod
    def load_vocabulary_mapping(vocab_path: str) -> dict[int, str]:
        print(vocab_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)

        id_to_string = {
            int(token_id): token_str
            for token_str, token_id in vocab_json.items()
        }
        return id_to_string

    def get_padded_ids(self, prompts: List[Prompt]) -> List[List[int]]:
        full_ids = []
        for prompt in prompts:
            full_prompt = PromptMaker.build_prompt(
                prompt,
                self.functions,
                self.model_path + "tokenizer_config.json",
            )
            full_ids.append(self.model.encode(full_prompt)[0].tolist())
        max_len = max(len(ids) for ids in full_ids)
        padded_ids = []
        for ids in full_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([self.eos_token] * pad_len + ids)
        return padded_ids

    def generate_with_constraints_batched(
        self, padded_ids: List[List[int]]
    ) -> List[str]:
        return self.generator.generate_batch(padded_ids)
