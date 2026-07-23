from typing import List
from pydantic import BaseModel
from src.models import FuncDefinition


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
            str: A formatted block mapping functions to their descriptions
            and parameters.
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
        Wrap the user's intent and the tools schema inside standard chat
        templates.

        Args:
            user_prompt (str): The query input by the user.
            functions (List[FuncDefinition]): List of allowed functions.

        Returns:
            str: The final templated prompt formatted for optimal
            instruction following.
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
