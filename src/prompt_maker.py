import json
from jinja2 import Template
from typing import List
from src.models import FuncDefinition


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

        return formatted_prompt
