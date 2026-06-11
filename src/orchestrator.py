import json
import sys
import numpy
from typing import List, Dict
from llm_sdk import Small_LLM_Model
from src.models import FuncDefinition


class Orchestrator:
    def __init__(self, functions: List) -> None:
        self.functions = functions
        self.model = Small_LLM_Model()
        self.vocab = self.load_vocabulary_mapping(
            self.model.get_path_to_vocab_file()
        )
        self.tokenizer = self.load_tokenizer(
            self.model.get_path_to_tokenizer_file()
        )
        self.constrainer = OutputConstrainer(self.vocab)

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> dict[str, str]:
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)

        return tokenizer

    @staticmethod
    def load_vocabulary_mapping(vocab_path: str) -> dict[int, str]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)

        id_to_string = {
            int(token_id): token_str
            for token_str, token_id in vocab_json.items()
        }
        return id_to_string

    def generate_with_constraints(self, user_prompt: str) -> str:
        print(f"Starting generation process with prompt: {user_prompt}")
        full_prompt = PromptMaker.build_prompt(user_prompt, self.functions)
        generator = Generator(self.model, self.tokenizer)
        answer_ids = []
        for token in generator.generate(full_prompt, self.constrainer):
            answer_ids.append(token)
        print("Decoding...")
        answer = self.model.decode(answer_ids)
        print(answer)


class Generator:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, constrainer) -> int:
        input_ids = self.model.encode(prompt)
        input_ids = input_ids.tolist()

        end_of_text_token = self.tokenizer["added_tokens"][0]["id"]

        generated_token = ""
        generated_text = ""
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids[0])
            logits_np = numpy.array(logits)
            logits_filtered = constrainer.apply_constraints(
                generated_text, logits_np
            )
           #print([logit for logit in logits_np if logit != -numpy.inf])
           #print([logit for logit in logits_filtered if logit != -numpy.inf])
            generated_token = int(numpy.argmax(logits_filtered))
           #sys.exit(1)
            if generated_token == end_of_text_token:
                break
            input_ids[0].append(generated_token)
            generated_text += self.model.decode(generated_token)
            yield generated_token


class OutputConstrainer:
    def __init__(self, vocab: Dict[int, str]) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def apply_constraints(
        self, current_text: str, logits_np: numpy.ndarray
    ) -> numpy.ndarray:
        mask = numpy.zeros(len(logits_np), dtype=bool)
        return_np = logits_np

        if len(current_text) == 0:
            for token_id, token_str in self.vocab.items():
                if "{" == token_str:
                    mask[token_id] = True
        else:
            mask[:] = True

        return_np[~mask] = -(numpy.inf)
        return return_np


class PromptMaker:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_functions_as_prompt(functions: List[FuncDefinition]) -> str:
        formatted = ""

        formatted += "Available functions\n"
        for func in functions:
            formatted += f"- {func.name}: {func.description}\n"
            formatted += "  Parameters:\n"
            for param_name, param_type_obj in func.parameters.items():
                formatted += f"     - {param_name} ({param_type_obj.type})\n"

        formatted += "\n"
        return formatted

    @staticmethod
    def build_prompt(user_prompt: str, functions: List[FuncDefinition]) -> str:
        final = ""

        final += "<|im_start|>system\n"
        final += "You are an expert that translates natural language requests"
        final += " into strict JSON function calls.\n"
        final += "You must NOT answer the question directly. You must ONLY"
        final += (
            " output a valid JSON object choosing the correct function.\n\n"
        )

        final += PromptMaker.get_functions_as_prompt(functions)
        final += "<|im_end|>\n"

        final += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        final += "<|im_start|>assistant\n"

        return final
