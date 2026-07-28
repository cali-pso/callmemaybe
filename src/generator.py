import numpy
from typing import List, Dict, Tuple, Optional, Any, Generator
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.models import FuncDefinition
from src.errors import UnknownFunctionError


MAX_TOKENS = 50


class GeneratorCls(BaseModel):
    pass
