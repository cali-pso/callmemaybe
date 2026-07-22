import numpy
from typing import List, Dict, Tuple, Optional, Any, Generator
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.models import FuncDefinition
from src.errors import UnknownFunctionError


MAX_TOKENS = 50


class GeneratorCls(BaseModel):
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
    def build_functions_dict(self) -> "GeneratorCls":
        """
        Map available function definitions by their name for rapid lookup.

        Returns:
            GeneratorCls: The updated instance.
        """
        self.functions_dict = {f.name: f for f in self.functions}
        return self

    def generate(
        self, prompt: str, constrainer: Any
    ) -> Generator[int, None, None]:
        """
        Yield valid tokens iteratively until max length or completion is
        reached.

        Args:
            prompt (str): The complete formatted prompt text.
            constrainer: OutputConstrainer object to enforce logit rules.

        Yields:
            int: Valid token IDs generated one by one.
        """
        input_ids = self.model.encode(prompt)
        input_ids = input_ids.tolist()

        end_of_text_token = self.tokenizer["added_tokens"][0]["id"]

        generated_text = ""

        json_has_started = False
        brace_count = 0
        tokens_count = 0
        context: Optional[Tuple[Optional[str], List[str], bool]] = None

        while tokens_count <= MAX_TOKENS:
            skip = self.check_for_skip(generated_text, brace_count, context)
            skipped_str = ""
            if skip is not None:
                for token in skip:
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
            try:
                logits_filtered, context = constrainer.apply_constraints(
                    generated_text, logits_np
                )
            except UnknownFunctionError as e:
                raise e
            else:
                func_name, current_param, is_inside_string = context
                generated_token_id = int(numpy.argmax(logits_filtered))
                if generated_token_id == end_of_text_token:
                    break
                generated_str = self.model.decode(generated_token_id)

                if brace_count != 0:
                    generated_text += str(skipped_str) + str(generated_str)
                    input_ids[0].append(generated_token_id)

                    if "{" in generated_str:
                        brace_count += 1
                        if not json_has_started:
                            json_has_started = True
                    if "}" in generated_str:
                        brace_count -= 1

                    tokens_count += 1
                    print(generated_str, end="", flush=True)
                    yield generated_token_id

                if brace_count == 0 and json_has_started:
                    break

    def check_for_skip(
        self,
        text: str,
        brace_count: int,
        context: Optional[Tuple[Optional[str], List[str], bool]],
    ) -> Any:
        """
        Determine if boilerplate JSON structuring can be safely fast-forwarded.

        Args:
            text (str): The current generated text string.
            brace_count (int): Counter for open curly brackets.
            context (Optional[Tuple]): Tuple containing function context.

        Returns:
            Optional[List[int]]: Encoded token list of the injected
            boilerplate, or None.
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

        if len(skipped) > 0:
            encoded = self.model.encode(skipped)
            if hasattr(encoded, "tolist"):
                return encoded.tolist()[0]
            return list(encoded)[0]
        return None
