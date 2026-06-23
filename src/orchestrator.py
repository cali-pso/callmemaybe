import json
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
        self.constrainer = OutputConstrainer(self.vocab, self.model)

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
        print(f"\n\nStarting generation process with {user_prompt}\n")
        full_prompt = PromptMaker.build_prompt(user_prompt, self.functions)
        generator = Generator(self.model, self.tokenizer)
        answer_ids = []
        for token in generator.generate(full_prompt, self.constrainer):
            answer_ids.append(token)
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

        json_has_started = False
        brace_count = 0
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids[0])
            logits_np = numpy.array(logits)
            logits_filtered = constrainer.apply_constraints(
                generated_text, logits_np
            )
            generated_token = int(numpy.argmax(logits_filtered))
            if generated_token == end_of_text_token:
                break
            input_ids[0].append(generated_token)
            generated_str = self.model.decode(generated_token)
            generated_text += generated_str

            if "{" in generated_str:
                brace_count += 1
                if not json_has_started:
                    json_has_started = True
            if "}" in generated_str:
                brace_count -= 1

            yield generated_token

            if brace_count == 0 and json_has_started:
                break


class OutputConstrainer:
    def __init__(self, vocab: Dict[int, str], model) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.model = model

        self.banned_tokens = set()
        banned_str_list = ["`", "<think>", "</think>"]
        for token_id, token_str in self.vocab.items():
            for s in banned_str_list:
                if s in token_str:
                    self.banned_tokens.add(token_id)

    def apply_constraints(
        self, current_text: str, logits_np: numpy.ndarray
    ) -> numpy.ndarray:
        mask = numpy.zeros(len(logits_np), dtype=bool)
        return_np = logits_np

        if len(current_text) == 0:
            for token_id, token_str in self.vocab.items():
                if "{" == token_str:
                    mask[token_id] = True
        elif current_text[-1] in ["{", "}"]:
            mask[self.model.encode("\n")] = True
        else:
            mask[:] = True
            for token in self.banned_tokens:
                mask[token] = False

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
