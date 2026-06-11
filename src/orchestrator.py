import json
import numpy
from typing import List
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
        for token in generator.generate(full_prompt):
            answer_ids.append(token)
        print("Decoding...")
        answer = self.model.decode(answer_ids)
        print(answer)


class Generator:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> int:
        input_ids = self.model.encode(prompt)
        input_ids = input_ids.tolist()

        end_of_text_token = self.tokenizer["added_tokens"][0]["id"]

        generated_token = ""
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids[0])
            logits_np = numpy.array(logits)
            generated_token = int(numpy.argmax(logits_np))
            if generated_token == end_of_text_token:
                break
            input_ids[0].append(generated_token)
            yield generated_token


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
        final += "<|im_start|>assistant\n{"

        return final


class OutputConstrainer:
    def __init__(self) -> None:
        pass

    def apply_constraints(self) -> None:
        pass
