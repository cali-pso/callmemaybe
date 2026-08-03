import re
import numpy
from typing import List, Dict, Any
from src.models import FuncDefinition
from src.errors import UnknownFunctionError

MAX_TOKENS = 250


class OutputConstrainer:
    def __init__(
        self,
        vocab: Dict[int, str],
        model: Any,
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
                        match = re.search(r'"name":\s*"([^"]+)"', full_text)
                        if match:
                            func_name = match.group(1)
                            self.function[i] = self.functions[func_name]
                        else:
                            raise UnknownFunctionError()
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
        self.state: List[str] = ["STATE_EXPECT_START"] * batch_size
        self.brackets: List[int] = [0] * batch_size
        self.quotes: List[int] = [0] * batch_size
        self.done_func_name: List[bool] = [False] * batch_size
        self.done_params: List[int] = [0] * batch_size
        self.in_params: List[int] = [0] * batch_size
        self.function: List[Any] = [None] * batch_size
        self.function_params: List[Any] = [None] * batch_size

    def create_masks(self) -> Dict[str, numpy.ndarray]:
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
        self, current_text: str, logits: numpy.ndarray, batch_idx: int
    ) -> numpy.ndarray:
        return_np = logits
        return_np[~self.masks[self.state[batch_idx]]] = -(numpy.inf)
        return return_np
