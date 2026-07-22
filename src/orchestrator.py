import json
from typing import List, Dict, Optional, Any
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import FuncDefinition
from src.errors import UnknownFunctionError
from src.prompt_maker import PromptMaker
from src.output_constrainer import OutputConstrainer
from src.generator import GeneratorCls


class Orchestrator(BaseModel):
    """
    Manages the end-to-end execution of constrained text generation.

    This class coordinates the Hugging Face model, its vocabulary, and the
    output masking logic to ensure the generated text tightly adheres to the
    expected JSON function schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"

    functions: List[FuncDefinition]

    model: Any = Field(default=None, init=False)
    vocab: Dict[int, str] = Field(default_factory=dict, init=False)
    tokenizer: Dict[str, Any] = Field(default_factory=dict, init=False)
    constrainer: Optional["OutputConstrainer"] = Field(
        default=None, init=False
    )

    @model_validator(mode="after")
    def initialize_components(self) -> "Orchestrator":
        """
        Post-initialization hook to load tokenizer resources and build
        the constrainer.

        Returns:
            Orchestrator: The initialized instance.
        """
        self.model = Small_LLM_Model(model_name=self.model_name)
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
    def load_tokenizer(tokenizer_path: str) -> Any:
        """
        Parse the JSON tokenizer configuration into a Python dictionary.

        Args:
            tokenizer_path (str): The absolute path to `tokenizer.json`.

        Returns:
            Dict[str, Any]: The parsed tokenizer object.
        """
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)

        return tokenizer

    @staticmethod
    def load_vocabulary_mapping(vocab_path: str) -> Dict[int, str]:
        """
        Build an integer-to-string vocabulary dictionary from the model
        vocab.

        Args:
            vocab_path (str): The absolute path to `vocab.json`.

        Returns:
            Dict[int, str]: A dictionary where keys are token IDs and
            values are tokens.
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
        Translate a natural language prompt into a strictly formatted JSON
        function call.

        Args:
            user_prompt (str): The input request from the user.

        Returns:
            str: The final, valid JSON string containing function name
            and arguments.
        """
        full_prompt = PromptMaker.build_prompt(user_prompt, self.functions)
        generator = GeneratorCls(
            model=self.model,
            tokenizer=self.tokenizer,
            functions=self.functions,
        )
        answer_ids = []
        try:
            for token in generator.generate(full_prompt, self.constrainer):
                answer_ids.append(token)
            answer = self.model.decode(answer_ids)
        except UnknownFunctionError:
            return str(
                '{\n\t"message":"Needed function was not found by model"\n}'
            )
        return str(answer)
