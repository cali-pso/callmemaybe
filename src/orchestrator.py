import json
import numpy
import re
from typing import List, Dict, Tuple, Optional, Any
from llm_sdk import Small_LLM_Model
from src.models import FuncDefinition

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TOKENS = 50


class Orchestrator(BaseModel):
    """
    Manages the end-to-end execution of constrained text generation.

    This class coordinates the Hugging Face model, its vocabulary, and the
    output masking logic to ensure the generated text tightly adheres to the
    expected JSON function schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: List[FuncDefinition]

    model: Any = Field(default_factory=Small_LLM_Model)
    vocab: Dict[int, str] = Field(default_factory=dict, init=False)
    tokenizer: Dict[str, Any] = Field(default_factory=dict, init=False)
    constrainer: Optional["OutputConstrainer"] = Field(
        default=None, init=False
    )

    @model_validator(mode="after")
    def initialize_components(self) -> "Orchestrator":
        """
        Post-initialization hook to load tokenizer resources and build the constrainer.

        Returns:
            Orchestrator: The initialized instance.
        """
        self.vocab = self.load_vocabulary_mapping(
            self.model.get_path_to_vocab_file()
        )
        self.tokenizer = self.load_tokenizer(
            self.model.get_path_to_tokenizer_file()
        )
        self.constrainer = OutputConstrainer(
            vocab=self.vocab, model=self.model, functions=self.functions
        )
        return self

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> dict[str, str]:
        """
        Parse the JSON tokenizer configuration into a Python dictionary.

        Args:
            tokenizer_path (str): The absolute path to `tokenizer.json`.

        Returns:
            dict[str, str]: The parsed tokenizer object.
        """
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)

        return tokenizer

    @staticmethod
    def load_vocabulary_mapping(vocab_path: str) -> dict[int, str]:
        """
        Build an integer-to-string vocabulary dictionary from the model vocab.

        Args:
            vocab_path (str): The absolute path to `vocab.json`.

        Returns:
            dict[int, str]: A dictionary where keys are token IDs and values are tokens.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)

        id_to_string = {
            int(token_id): token_str
            for token_str, token_id in vocab_json.items()
        }
        return id_to_string

    def generate_with_constraints(self, user_prompt: str) -> str:
        """
        Translate a natural language prompt into a strictly formatted JSON function call.

        Args:
            user_prompt (str): The input request from the user.

        Returns:
            str: The final, valid JSON string containing function name and arguments.
        """
        full_prompt = PromptMaker.build_prompt(user_prompt, self.functions)
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            functions=self.functions,
        )
        answer_ids = []
        for token in generator.generate(full_prompt, self.constrainer):
            answer_ids.append(token)
        answer = self.model.decode(answer_ids)
        return answer


class Generator(BaseModel):
    """
    Handles the token-by-token autoregressive decoding process.

    Implements a controlled generation loop that queries logit outputs and
    applies logical masking via an output constrainer before selecting the
    next valid token.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    tokenizer: Dict[str, Any]
    functions: List[FuncDefinition]

    functions_dict: Dict[str, FuncDefinition] = Field(
        default_factory=dict, init=False
    )

    @model_validator(mode="after")
    def build_functions_dict(self) -> "Generator":
        """
        Map available function definitions by their name for rapid lookup.

        Returns:
            Generator: The updated instance.
        """
        self.functions_dict = {f.name: f for f in self.functions}
        return self

    def generate(self, prompt: str, constrainer) -> int:
        """
        Yield valid tokens iteratively until max length or completion is reached.

        Args:
            prompt (str): The complete formatted prompt text.
            constrainer: OutputConstrainer object to enforce logit rules.

        Yields:
            int: Valid token IDs generated one by one.
        """
        input_ids = self.model.encode(prompt)
        input_ids = input_ids.tolist()

        end_of_text_token = self.tokenizer["added_tokens"][0]["id"]

        generated_token = ""
        generated_text = ""

        json_has_started = False
        brace_count = 0
        tokens_count = 0
        context = None
        while tokens_count <= MAX_TOKENS:
            skip = self.check_for_skip(generated_text, brace_count, context)
            skipped_str = ""
            if skip is not None:
                for token in skip[0]:
                    input_ids[0].append(token)
                    decoded = self.model.decode(token)
                    if "{" in decoded:
                        brace_count += 1
                        if not json_has_started:
                            json_has_started = True
                    if "}" in decoded:
                        brace_count -= 1
                    skipped_str += decoded
                    yield token

            logits = self.model.get_logits_from_input_ids(input_ids[0])
            logits_np = numpy.array(logits)
            logits_filtered, context = constrainer.apply_constraints(
                generated_text, logits_np
            )
            func_name, current_param, is_inside_string = context
            generated_token = int(numpy.argmax(logits_filtered))
            if generated_token == end_of_text_token:
                break
            generated_str = self.model.decode(generated_token)

            if brace_count != 0:
                generated_text += str(skipped_str) + str(generated_str)
                input_ids[0].append(generated_token)

                if "{" in generated_str:
                    brace_count += 1
                    if not json_has_started:
                        json_has_started = True
                if "}" in generated_str:
                    brace_count -= 1
                skipped_str += decoded

                tokens_count += 1
                print(generated_str, end="", flush=True)
                yield generated_token

            if brace_count == 0 and json_has_started:
                break

    def check_for_skip(
        self, text: str, brace_count: int, context: Tuple
    ) -> list[int]:
        """
        Determine if boilerplate JSON structuring can be safely fast-forwarded.

        Args:
            text (str): The current generated text string.
            brace_count (int): Counter for open curly brackets.
            context (Tuple): Tuple containing the function context metadata.

        Returns:
            list[int]: Encoded token list of the injected boilerplate, or None.
        """
        skipped = ""
        if not text:
            skipped += '{\n"func_name": "'
        if context is not None:
            func_name, params_match, is_inside_string = context
            if func_name and '"parameters": {\n' not in text:
                skipped += 'parameters": {\n'
                text += skipped
            if func_name and "parameters" in text and text[-1] == "\n":
                func_def = self.functions_dict[func_name]
                param_names = list(func_def.parameters.keys())
                if len(param_names) > len(params_match):
                    if param_names[len(params_match)] not in params_match:
                        skipped += f'  "{param_names[len(params_match)]}": '
                else:
                    skipped += "}\n}\n"

        tokens = self.model.encode(skipped) if len(skipped) > 0 else None
        return tokens


class OutputConstrainer(BaseModel):
    """
    Applies logic-based mask filtering to probability logits.

    By parsing the in-progress JSON string, this class determines which data type
    is currently required and forcibly sets logits for invalid tokens to negative
    infinity, guaranteeing the model outputs syntax-compliant values.
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
    essential_tokens: Dict[str, int] = Field(default_factory=dict, init=False)
    default_mask: numpy.ndarray = Field(default=None, init=False)
    int_mask: numpy.ndarray = Field(default=None, init=False)
    float_mask: numpy.ndarray = Field(default=None, init=False)
    string_mask: numpy.ndarray = Field(default=None, init=False)

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

    def _build_char_subset_mask(self, allowed_chars: str) -> numpy.ndarray:
        """
        Creates a mask where tokens are only allowed if ALL their characters
        belong to the allowed_chars string.
        """
        mask = self.default_mask.copy()
        allowed_set = set(allowed_chars)
        for token_id, token_str in self.vocab.items():
            # Strip spaces to ensure we don't accidentally block spacing tokens
            if set(token_str.strip()).issubset(allowed_set):
                mask[token_id] = True
        return mask

    def _build_safe_string_mask(self) -> numpy.ndarray:
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
    ) -> Tuple[Optional[str], Optional[str], bool]:
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
    ) -> numpy.ndarray:
        """
        Dynamically filters invalid token logits based on the current parsed state.

        Args:
            current_text (str): The JSON text generated so far.
            logits_np (numpy.ndarray): The raw 1D array of unnormalized token scores.

        Returns:
            numpy.ndarray: The modified logit array where invalid tokens are heavily penalized.
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
        return (return_np, context)


class PromptMaker(BaseModel):
    """
    Utility class dedicated to assembling instructions and schemas.
    """

    @staticmethod
    def get_functions_as_prompt(functions: List[FuncDefinition]) -> str:
        """
        Serialize available function metadata into a readable string block.

        Args:
            functions (List[FuncDefinition]): List of functions to format.

        Returns:
            str: A formatted block mapping functions to their descriptions and parameters.
        """
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
        """
        Wrap the user's intent and the tools schema inside standard chat templates.

        Args:
            user_prompt (str): The query input by the user.
            functions (List[FuncDefinition]): List of allowed functions.

        Returns:
            str: The final templated prompt formatted for optimal instruction following.
        """
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
