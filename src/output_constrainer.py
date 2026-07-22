import re
import numpy
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.models import FuncDefinition
from src.errors import UnknownFunctionError


class OutputConstrainer(BaseModel):
    """
    Applies logic-based mask filtering to probability logits.

    By parsing the in-progress JSON string, this class determines which data
    type is currently required and forcibly sets logits for invalid tokens to
    negative infinity, guaranteeing syntax-compliant outputs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab: Dict[int, str]
    model: Any  # Allowing Small_LLM_Model
    functions: List[FuncDefinition]

    # Fields generated after initialization
    vocab_size: int = Field(default=0, init=False)
    functions_dict: Dict[str, FuncDefinition] = Field(
        default_factory=dict, init=False
    )
    essential_tokens: Dict[str, Any] = Field(default_factory=dict, init=False)
    default_mask: Any = Field(default=None, init=False)
    int_mask: Any = Field(default=None, init=False)
    float_mask: Any = Field(default=None, init=False)
    string_mask: Any = Field(default=None, init=False)

    @model_validator(mode="after")
    def setup_masks_and_dicts(self) -> "OutputConstrainer":
        """
        Sets up required attributes
        """
        self.vocab_size = len(self.vocab)
        self.functions_dict = {f.name: f for f in self.functions}

        banned_str_list = ["`", "<think>", "</think>", ' "', " {", " }"]
        needed_tokens_str = ["{", "}", "\\"]

        self.essential_tokens = {
            c: self.model.encode(c) for c in needed_tokens_str
        }

        # Initialize masks
        self.default_mask = numpy.ones(151936, dtype=bool)
        for token_id, token_str in self.vocab.items():
            for s in banned_str_list:
                if s in token_str:
                    self.default_mask[token_id] = False

        self.int_mask = self._build_char_subset_mask("0123456789,\n")
        self.float_mask = self._build_char_subset_mask("0123456789.,\n")
        self.string_mask = self._build_safe_string_mask()

        return self

    def _build_char_subset_mask(self, allowed_chars: str) -> Any:
        """
        Creates a mask where tokens are only allowed if ALL their characters
        belong to the allowed_chars string.
        """
        mask = self.default_mask.copy()
        allowed_set = set(allowed_chars)
        for token_id, token_str in self.vocab.items():
            # Strip spaces to ensure we don't accidentally block spacing
            if set(token_str.strip()).issubset(allowed_set):
                mask[token_id] = True
        return mask

    def _build_safe_string_mask(self) -> Any:
        """
        Bans tokens that contain unescaped quotes or newlines, preventing
        the LLM from breaking JSON syntax internally.
        """
        mask = self.default_mask.copy()
        # for token_id, token_str in self.vocab.items():
        #   if "{" in token_str or "}" in token_str and "\\" not in token_str:
        #       mask[token_id] = False
        return mask

    def _get_current_context(
        self, current_text: str
    ) -> Tuple[Optional[str], List[str], bool]:
        """
        Parses the generated JSON text to determine:
        1. Function Name
        2. Current Parameter
        3. Whether the cursor is currently inside a string value
        """
        name_match = re.search(r'"func_name"\s*:\s*"([^"]+)"', current_text)
        func_name = name_match.group(1) if name_match else None

        params_match = re.findall(r'"([^"]+)"\s*:', current_text)
        params_match = [
            s for s in params_match if s not in ["func_name", "parameters"]
        ]

        text_no_escapes = current_text.replace('\\"', "")
        is_inside_string = text_no_escapes.count('"') % 2 != 0

        return func_name, params_match, is_inside_string

    def apply_constraints(
        self, current_text: str, logits_np: numpy.ndarray
    ) -> Tuple[numpy.ndarray, Tuple[Optional[str], List[str], bool]]:
        """
        Dynamically filters invalid token logits based on the current
        parsed state.

        Args:
            current_text (str): The JSON text generated so far.
            logits_np (numpy.ndarray): The raw 1D array of unnormalized
                token scores.

        Returns:
            Tuple[numpy.ndarray, Tuple[Optional[str], List[str], bool]]:
                The modified logit array where invalid tokens are heavily
                penalized, along with the parsed context.
        """
        return_np = logits_np
        context = self._get_current_context(current_text)
        func_name, params_match, is_inside_string = context
        current_param = params_match[-1] if params_match else None

        if func_name in self.functions_dict and current_param:
            func_def = self.functions_dict[func_name]

            if current_param in func_def.parameters:
                param_type = func_def.parameters[current_param].type.lower()

                if f'"{current_param}": ' in current_text:
                    if param_type == "int" and not is_inside_string:
                        return_np[~self.int_mask] = -(numpy.inf)
                        return (return_np, context)

                    elif param_type == "float" and not is_inside_string:
                        return_np[~self.float_mask] = -(numpy.inf)
                        return (return_np, context)

                    elif param_type == "string" and is_inside_string:
                        return_np[~self.string_mask] = -(numpy.inf)
                        return (return_np, context)
        elif func_name and func_name not in self.functions_dict:
            raise UnknownFunctionError(func_name)
        return (return_np, context)
