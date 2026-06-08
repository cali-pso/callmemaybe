from typing import Dict
from pydantic import BaseModel


class Prompt(BaseModel):
    """Represents a prompt as defined in input JSON"""

    prompt: str


class ParameterType(BaseModel):
    """Represents a specific type for a parameter"""

    type: str


class FuncDefinition(BaseModel):
    """Represents a full function definition"""

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType
