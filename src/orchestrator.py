import json
import re
import numpy
import os
import torch
import time
from typing import List, Dict
from llm_sdk import Small_LLM_Model
from src.models import FuncDefinition, Prompt
from src.errors import UnknownFunctionError
from jinja2 import Template

MAX_TOKENS = 250


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
    def get_model_path(model: Small_LLM_Model) -> str:
        tokenizer_path = model.get_path_to_tokenizer_file()
        return os.path.dirname(tokenizer_path) + "/"

    def load_tokenizer(self, tokenizer_path: str) -> dict[str, str]:
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

    def generate_with_constraints(self, user_prompt: str) -> str:
        print(f"\n\nStarting generation process with {user_prompt}\n")
        full_prompt = PromptMaker.build_prompt(
            user_prompt,
            self.functions,
            self.model_path + "tokenizer_config.json",
        )
        answer_trad = ""
        for choosen_trad in self.generator.generate(full_prompt):
            answer_trad += choosen_trad
            print(choosen_trad, end="", flush=True)
        return answer_trad


class Generator:
    def __init__(self, model, tokenizer, constrainer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.constrainer = constrainer
        self.eos_token = self.tokenizer["added_tokens"][0]["id"]
        self.start_time = time.time()
        self.last_observed_time = self.start_time

    def display_generated_texts(self, generated_texts: List[str]) -> None:
        print("-----------------------------------------------------------\n")
        for text in generated_texts:
            print(text)
        print()
        cur_time = time.time()
        gen_time = cur_time - self.last_observed_time
        full_time = cur_time - self.start_time
        print(
            f"Time since generation start: {full_time:4f}s |",
            f"Batch generation time: {gen_time:4f}s",
            f"| {(gen_time / len(generated_texts)):4f} T/s\n",
        )
        self.last_observed_time = cur_time

    def generate_batch(self, input_ids: List[List[int]]) -> List[str]:
        batch_size = len(input_ids)
        generated_texts = [""] * batch_size
        active_indices = list(range(batch_size))
        tokens_count = 0
        past_key_values = None

        input_tensor = torch.tensor(
            input_ids, device=self.model._device, dtype=torch.long
        )
        attention_mask = (input_tensor != self.eos_token).long()
        self.constrainer.reset_state(batch_size=batch_size)

        while tokens_count <= MAX_TOKENS and len(active_indices) > 0:
            self.display_generated_texts(generated_texts)
            with torch.no_grad():
                out = self.model._model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            next_tokens = []
            next_active_indices = []
            rows_to_keep = []

            for current_idx, original_idx in enumerate(active_indices):
                if self.constrainer.state[original_idx] != "STATE_FINISHED":
                    seq_logits = next_token_logits[current_idx].tolist()
                    logits_filtered = self.constrainer.apply_constraints(
                        generated_texts[original_idx],
                        seq_logits,
                        batch_idx=original_idx,
                    )
                    chosen_token = int(numpy.argmax(logits_filtered))
                else:
                    chosen_token = self.eos_token
                next_tokens.append(chosen_token)
                chosen_trad = (
                    self.constrainer.vocab[chosen_token].replace("Ġ", " ")
                    if chosen_token != self.eos_token
                    else ""
                )
                generated_texts[original_idx] += chosen_trad

                if (
                    chosen_token == self.eos_token
                    or self.constrainer.state[original_idx] == "STATE_FINISHED"
                ):
                    pass
                else:
                    self.constrainer.update_state(
                        chosen_trad,
                        generated_texts[original_idx],
                        batch_idx=original_idx,
                    )
                    next_active_indices.append(original_idx)
                    rows_to_keep.append(current_idx)

            if not next_active_indices:
                break

            input_tensor = input_tensor[rows_to_keep]

            if past_key_values is not None:
                if hasattr(past_key_values, "reorder_cache"):
                    # For modern Hugging Face models using Cache objects
                    keep_indices = torch.tensor(
                        rows_to_keep,
                        device=self.model._device,
                        dtype=torch.long,
                    )
                    past_key_values.reorder_cache(keep_indices)
                else:
                    # Fallback for legacy tuple formats
                    past_key_values = tuple(
                        tuple(
                            tensor[rows_to_keep]
                            if tensor is not None
                            else None
                            for tensor in layer_cache
                        )
                        for layer_cache in past_key_values
                    )
            attention_mask = attention_mask[rows_to_keep]
            new_mask_col = torch.ones(
                (len(rows_to_keep), 1),
                device=self.model._device,
                dtype=torch.long,
            )
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            active_next_tokens = [next_tokens[i] for i in rows_to_keep]
            input_tensor = torch.tensor(
                active_next_tokens, device=self.model._device, dtype=torch.long
            ).unsqueeze(1)
            active_indices = next_active_indices
            tokens_count += 1

        return generated_texts

    def generate(self, prompt: str) -> int:

        tokens_count = 0
        generated_text = ""
        choosen_token = None
        input_ids = self.model.encode(prompt)[0].tolist()

        self.constrainer.reset_state()
        while (
            tokens_count <= MAX_TOKENS
            and self.constrainer.state != "STATE_FINISHED"
        ):
            if choosen_token is not None:
                input_ids.append(choosen_token)

            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_filtered = self.constrainer.apply_constraints(
                generated_text, logits
            )

            choosen_token = numpy.argmax(logits_filtered)
            choosen_trad = self.constrainer.vocab[int(choosen_token)].replace(
                "Ġ", " "
            )

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
        self.masks = self.create_masks()

    def update_state(
        self, new_text: str, full_text: str, batch_idx: int
    ) -> None:
        i = batch_idx
        try:
            if (
                self.state[i] == "STATE_EXPECT_START"
                and "{" in new_text
                and self.brackets[i] == 0
            ):
                self.brackets[i] += 1
                self.state[i] = "STATE_EXPECT_FN_NAME_KEY"

            elif self.state[i] == "STATE_EXPECT_FN_NAME_KEY":
                if '"' in new_text:
                    self.quotes[i] += new_text.count('"')
                if self.quotes[i] == 2:
                    self.quotes[i] = 0
                    self.state[i] = "STATE_EXPECT_COLON"

            elif (
                self.state[i] == "STATE_EXPECT_COLON"
                and ":" in new_text
                and not self.done_func_name[i]
            ):
                self.state[i] = "STATE_EXPECT_FN_NAME_VALUE"

            elif self.state[i] == "STATE_EXPECT_FN_NAME_VALUE":
                if '"' in new_text:
                    self.quotes[i] += new_text.count('"')
                if self.quotes[i] == 2:
                    self.quotes[i] = 0
                    try:
                        self.function[i] = self.functions[full_text.split('"')[3]]
                        self.function_params[i] = list(
                            self.function[i].parameters.values()
                        )
                    except Exception:
                        raise UnknownFunctionError()
                    self.state[i] = "STATE_EXPECT_COMMA"

            elif (
                self.state[i] == "STATE_EXPECT_COMMA"
                and "," in new_text
                and not self.done_func_name[i]
            ):
                self.done_func_name[i] = True
                self.state[i] = "STATE_EXPECT_PARAM_KEY"

            elif self.state[i] == "STATE_EXPECT_PARAM_KEY":
                if '"' in new_text:
                    self.quotes[i] += new_text.count('"')
                if self.quotes[i] == 2:
                    self.quotes[i] = 0
                    self.state[i] = "STATE_EXPECT_COLON"

            elif (
                self.state[i] == "STATE_EXPECT_COLON"
                and ":" in new_text
                and self.done_func_name[i]
                and not self.in_params[i]
            ):
                self.in_params[i] = True
                self.state[i] = "STATE_EXPECT_START"

            elif (
                self.state[i] == "STATE_EXPECT_START"
                and "{" in new_text
                and self.brackets[i] == 1
            ):
                self.brackets[i] += 1
                self.state[i] = "STATE_EXPECT_PARAM_NAME"

            elif self.state[i] == "STATE_EXPECT_PARAM_NAME":
                if '"' in new_text:
                    self.quotes[i] += new_text.count('"')
                if self.quotes[i] == 2:
                    self.quotes[i] = 0
                    self.state[i] = "STATE_EXPECT_COLON"

            elif self.state[i] == "STATE_EXPECT_COLON" and ":" in new_text:
                param_type = self.function_params[i][self.done_params[i]].type
                if param_type == "integer":
                    self.state[i] = "STATE_EXPECT_INT_PARAM"
                elif param_type == "number" or param_type == "float":
                    self.state[i] = "STATE_EXPECT_FLOAT_PARAM"
                elif param_type == "string":
                    self.state[i] = "STATE_EXPECT_STRING_PARAM"

            elif self.state[i] in [
                "STATE_EXPECT_INT_PARAM",
                "STATE_EXPECT_FLOAT_PARAM",
                "STATE_EXPECT_NUMBER_PARAM",
            ] and ("," in new_text or "}" in new_text):
                self.done_params[i] += 1
                if (
                    self.done_params[i] < len(self.function[i].parameters)
                    and "," in new_text
                ):
                    self.state[i] = "STATE_EXPECT_PARAM_NAME"
                elif "}" in new_text:
                    self.brackets[i] -= 1
                    self.state[i] = "STATE_EXPECT_END"

            elif self.state[i] == "STATE_EXPECT_STRING_PARAM":
                if '"' in new_text:
                    self.quotes[i] += new_text.count('"')
                if self.quotes[i] == 2 and (
                    "," in new_text or "}" in new_text or " " in new_text
                ):
                    self.quotes[i] = 0
                    self.done_params[i] += 1
                    if (
                        self.done_params[i] < len(self.function[i].parameters)
                        and "," in new_text
                    ):
                        self.state[i] = "STATE_EXPECT_PARAM_NAME"
                    elif "}" in new_text or " " in new_text:
                        if "}" in new_text:
                            self.brackets[i] -= 1
                        self.state[i] = "STATE_EXPECT_END"

            elif self.state[i] == "STATE_EXPECT_END":
                if "}" in new_text:
                    self.brackets[i] -= 1
                if self.brackets[i] == 0:
                    self.state[i] = "STATE_FINISHED"
        except UnknownFunctionError:
            self.state[i] = "STATE_EXPECT_END"

    def reset_state(self, batch_size: int) -> None:
        self.state = ["STATE_EXPECT_START"] * batch_size
        self.brackets = [0] * batch_size
        self.quotes = [0] * batch_size
        self.done_func_name = [False] * batch_size
        self.done_params = [0] * batch_size
        self.in_params = [0] * batch_size
        self.function = [None] * batch_size
        self.function_params = [None] * batch_size

    def create_masks(self) -> Dict[str, numpy.array]:
        start = self.get_mask(r"^[\{]$")
        colon = self.get_mask(r"^:\s?$")
        comma = self.get_mask(r"^[,]$")
        func_name_key = self.get_mask(r'^[\s"name]+$')
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
                if "\\" in token_str and "\\\\" not in token_str:
                    pass
                else:
                    mask[token_id] = True

        return mask

    def apply_constraints(
        self, current_text: str, logits: list[float], batch_idx: int
    ) -> list[float]:
        return_np = numpy.array(logits, dtype=numpy.float32)
        return_np[~self.masks[self.state[batch_idx]]] = -(numpy.inf)
        return return_np


class PromptMaker:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_functions_as_prompt(functions: List[FuncDefinition]) -> str:
        """
        Formats the available functions into a readable list for the LLM.
        """
        formatted = "Functions:"
        for func in functions:
            formatted += f"{func.name}:{func.description}|Args:"
            for param_name, param_type_obj in func.parameters.items():
                formatted += f"{param_name}({param_type_obj.type}),"
        return formatted

    @staticmethod
    def get_messages(
        user_prompt: str, functions: List[FuncDefinition]
    ) -> dict[str, str]:
        """
        Builds the system and user messages with a strict schema definition.
        """
        sys = "Route user requests to the appropriate function.\n"
        sys += "OUTPUT FORMAT:"
        sys += "{"
        sys += '"name": "function_name", '
        sys += '"parameters": {'
        sys += '"param1": "value1"'
        sys += "}"
        sys += "}\n"
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
