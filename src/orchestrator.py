import json
import re
import numpy
import os
from typing import List, Dict
from llm_sdk import Small_LLM_Model
from src.models import FuncDefinition
from jinja2 import Template

MAX_TOKENS = 50


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

    @staticmethod
    def get_model_path(model: Small_LLM_Model) -> str:
        tokenizer_path = model.get_path_to_tokenizer_file()
        return os.path.dirname(tokenizer_path) + "/"

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> dict[str, str]:
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)

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

    def generate_with_constraints(self, user_prompt: str) -> str:
        print(f"\n\nStarting generation process with {user_prompt}\n")
        full_prompt = PromptMaker.build_prompt(
            user_prompt,
            self.functions,
            self.model_path + "tokenizer_config.json",
        )
        generator = Generator(self.model, self.tokenizer, self.constrainer)
        answer_trad = []
        for choosen_trad in generator.generate(full_prompt):
            answer_trad.append(choosen_trad)
            print(choosen_trad, end="", flush=True)
        return answer_trad


class Generator:
    def __init__(self, model, tokenizer, constrainer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.constrainer = constrainer
        self.eos_token = self.tokenizer["added_tokens"][0]["id"]

    def generate(self, prompt: str) -> int:

        json_has_started = False
        brace_count = 0
        tokens_count = 0
        generated_text = ""
        choosen_token = None
        input_ids = self.model.encode(prompt)[0].tolist()

        while tokens_count <= MAX_TOKENS and self.constrainer.state != "STATE_FINISHED":
            if choosen_token is not None:
                input_ids.append(choosen_token)

            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_filtered = self.constrainer.apply_constraints(
                generated_text, logits
            )

            choosen_token = numpy.argmax(logits_filtered)
            choosen_trad = self.model.decode(choosen_token)

            generated_text += choosen_trad

            yield choosen_trad

            if choosen_token == self.eos_token:
                break
            else:
                self.constrainer.update_state(choosen_trad, generated_text)


class OutputConstrainer:
    def __init__(
        self,
        vocab: Dict[int, str],
        model: Small_LLM_Model,
        functions: List[FuncDefinition],
    ) -> None:
        self.functions = {function.name: function for function in functions}
        self.vocab = vocab
        self.vocab_size = len(model.get_logits_from_input_ids([0]))
        self.model = model
        self.state = "STATE_EXPECT_START"
        self.masks = self.create_masks()
        self.quotes = 0
        self.brackets = 0
        self.done_func_name = False
        self.in_params = False
        self.done_params = 0

    def update_state(self, new_text: str, full_text: str) -> None:

        if (
            self.state == "STATE_EXPECT_START"
            and "{" in new_text
            and self.brackets == 0
        ):
            self.brackets += 1
            self.state = "STATE_EXPECT_FN_NAME_KEY"

        elif self.state == "STATE_EXPECT_FN_NAME_KEY":
            if '"' in new_text:
                self.quotes += 1
            if self.quotes == 2:
                self.quotes = 0
                self.state = "STATE_EXPECT_COLON"

        elif (
            self.state == "STATE_EXPECT_COLON"
            and ":" in new_text
            and not self.done_func_name
        ):
            self.state = "STATE_EXPECT_FN_NAME_VALUE"

        elif self.state == "STATE_EXPECT_FN_NAME_VALUE":
            if '"' in new_text:
                self.quotes += 1
            if self.quotes == 2:
                self.quotes = 0
                self.state = "STATE_EXPECT_COMMA"

        elif (
            self.state == "STATE_EXPECT_COMMA"
            and "," in new_text
            and not self.done_func_name
        ):
            self.done_func_name = True
            self.state = "STATE_EXPECT_PARAM_KEY"

        elif self.state == "STATE_EXPECT_PARAM_KEY":
            if '"' in new_text:
                self.quotes += 1
            if self.quotes == 2:
                self.quotes = 0
                self.state = "STATE_EXPECT_COLON"

        elif (
            self.state == "STATE_EXPECT_COLON"
            and ":" in new_text
            and self.done_func_name
            and not self.in_params
        ):
            self.in_params = True
            self.state = "STATE_EXPECT_START"

        elif (
            self.state == "STATE_EXPECT_START"
            and "{" in new_text
            and self.brackets == 1
        ):
            self.brackets += 1
            self.state = "STATE_EXPECT_PARAM_NAME"

        elif self.state == "STATE_EXPECT_PARAM_NAME":
            if '"' in new_text:
                self.quotes += 1
            if self.quotes == 2:
                self.quotes = 0
                self.state = "STATE_EXPECT_COLON"

        elif self.state == "STATE_EXPECT_COLON" and ":" in new_text:
            self.function = self.functions[full_text.split('"')[3]]
            param_type = list(self.function.parameters.values())[
                self.done_params
            ].type
            if param_type == "number" or param_type == "int":
                self.state = "STATE_EXPECT_INT_PARAM"
            elif param_type == "float":
                self.state = "STATE_EXPECT_FLOAT_PARAM"
            elif param_type == "string":
                self.state = "STATE_EXPECT_STRING_PARAM"

        elif self.state in [
            "STATE_EXPECT_INT_PARAM",
            "STATE_EXPECT_FLOAT_PARAM",
            "STATE_EXPECT_NUMBER_PARAM",
        ] and ("," in new_text or "}" in new_text):
            self.done_params += 1
            if (
                self.done_params < len(self.function.parameters)
                and "," in new_text
            ):
                self.state = "STATE_EXPECT_PARAM_NAME"
            elif "}" in new_text:
                self.brackets -= 1
                self.state = "STATE_EXPECT_END"

        elif self.state == "STATE_EXPECT_STRING_PARAM":
            if '"' in new_text:
                self.quotes += 1
            if self.quotes == 2 and ("," in new_text or "}" in new_text):
                self.quotes = 0
                self.done_params += 1
                if self.done_params < len(self.function.parameters) and "," in new_text:
                    self.state = "STATE_EXPECT_PARAM_NAME"
                elif "}" in new_text:
                    self.brackets -= 1
                    self.state = "STATE_EXPECT_END"

        elif self.state == "STATE_EXPECT_END":
            if "}" in new_text:
                self.brackets -= 1
            if self.brackets == 0:
                self.state = "STATE_FINISHED"

    def create_masks(self) -> Dict[str, numpy.array]:
        start = self.get_mask(r"^\s?[\{]$")
        colon = self.get_mask(r"^:\s?$")
        comma = self.get_mask(r"^[,]$")
        func_name_key = self.get_mask(r'^[\s"func_ame]+$')
        func_name_val = self.get_mask(r'^[\s"A-Za-z0-9_]+$')
        param_key = self.get_mask(r'^[\s"parmetes]+$')
        int_param = self.get_mask(r"^[\s0-9,\}]+$")
        float_param = self.get_mask(r"^[\s0-9.,\}]+$")
        string_param = self.get_mask(
            r'^[\sA-Za-z0-9~`!@#\$%\^&\*\(\)_\+-=\{\}\[\]\|\\:;"\'<,>\.\?/]+$'
        )
        param_name = self.get_mask(r'^[\s"a-zA-Z0-9_]+$')
        end = self.get_mask(r"^[\}]$")

        masks = {
            "STATE_EXPECT_START": start,
            "STATE_EXPECT_COLON": colon,
            "STATE_EXPECT_COMMA": comma,
            "STATE_EXPECT_FN_NAME_KEY": func_name_key,
            "STATE_EXPECT_FN_NAME_VALUE": func_name_val,
            "STATE_EXPECT_PARAM_KEY": param_key,
            "STATE_EXPECT_INT_PARAM": int_param,
            "STATE_EXPECT_FLOAT_PARAM": float_param,
            "STATE_EXPECT_STRING_PARAM": string_param,
            "STATE_EXPECT_PARAM_NAME": param_name,
            "STATE_EXPECT_END": end,
        }
        return masks

    def get_mask(self, regex_pattern: str) -> numpy.ndarray:
        """
        Creates a boolean NumPy array where True indicates the token
        is built entirely from the allowed character set.
        """
        mask = numpy.zeros(self.vocab_size, dtype=bool)

        compiled_regex = re.compile(regex_pattern)

        for token_id_str, token_str in self.vocab.items():
            token_id = int(token_id_str)

            clean_token = token_str.replace("Ġ", " ")

            if compiled_regex.fullmatch(clean_token):
                mask[token_id] = True

        return mask

    def apply_constraints(
        self, current_text: str, logits: list[float]
    ) -> list[float]:
        return_np = numpy.asarray(logits)
        return_np[~self.masks[self.state]] = -(numpy.inf)
        return return_np


class PromptMaker:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_functions_as_prompt(functions: List[FuncDefinition]) -> str:
        """
        Formats the available functions into a clear, readable list for the LLM.
        """
        formatted = "AVAILABLE FUNCTIONS:\n"
        for func in functions:
            formatted += f"- Function: {func.name}\n"
            formatted += f"  Description: {func.description}\n"
            formatted += "  Parameters:\n"
            for param_name, param_type_obj in func.parameters.items():
                formatted += f"    * {param_name} ({param_type_obj.type})\n"
        formatted += "\n"
        return formatted

    @staticmethod
    def get_messages(
        user_prompt: str, functions: List[FuncDefinition]
    ) -> dict[str, str]:
        """
        Builds the system and user messages with a strict schema definition.
        """
        sys = "You are an intelligent API orchestrator. Your ONLY job is to route user requests to the appropriate function.\n\n"
        sys += "OUTPUT FORMAT:\n"
        sys += "You must respond ONLY with a raw JSON object matching this exact schema.\n"
        sys += "{ "
        sys += '"func_name": "the_name_of_the_function", '
        sys += '"parameters": { '
        sys += '"param1": "value1" '
        sys += "} "
        sys += "}\n\n"
        sys += PromptMaker.get_functions_as_prompt(functions)
        messages = [
            {"role": "system", "content": f"{sys}"},
            {"role": "user", "content": f"{user_prompt.prompt}"},
        ]
        return messages

    @staticmethod
    def build_prompt(
        user_prompt: str,
        functions: List[FuncDefinition],
        tokenizer_config_path: str,
    ) -> str:

        with open(tokenizer_config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        raw_template = config.get(
            "chat_template",
            "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}",
        )
        messages = PromptMaker.get_messages(user_prompt, functions)

        jinja_template = Template(raw_template)
        formatted_prompt = jinja_template.render(
            messages=messages, add_generation_prompt=True
        )

        print(formatted_prompt)
        return formatted_prompt
