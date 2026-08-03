from typing import Dict
from pydantic import BaseModel


class Prompt(BaseModel):
    """Represent a prompt as defined in input JSON."""

    prompt: str


class ParameterType(BaseModel):
    """Represent a specific type for a parameter."""

    type: str


class FuncDefinition(BaseModel):
    """Represent a full function definition."""

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType
