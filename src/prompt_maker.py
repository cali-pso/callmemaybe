import json
from jinja2 import Template
from typing import List, Optional
from src.models import FuncDefinition, Prompt


class PromptMaker:
    _cached_template: Optional[Template] = None

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
        user_prompt: Prompt, functions: List[FuncDefinition]
    ) -> list[dict[str, str]]:
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

    @classmethod
    def get_template(cls, tokenizer_config_path: str) -> Template:
        """Reads the file only if it hasn't been cached yet."""
        if cls._cached_template is None:
            with open(
                tokenizer_config_path, "r", encoding="utf-8"
            ) as config_file:
                config = json.load(config_file)
            raw_template = config.get("chat_template")
            cls._cached_template = Template(raw_template)
        return cls._cached_template

    @staticmethod
    def build_prompt(
        user_prompt: Prompt,
        functions: List[FuncDefinition],
        tokenizer_config_path: str,
    ) -> str:
        messages = PromptMaker.get_messages(user_prompt, functions)
        jinja_template = PromptMaker.get_template(tokenizer_config_path)

        formatted_prompt = jinja_template.render(
            messages=messages, add_generation_prompt=True
        )
        return formatted_prompt
