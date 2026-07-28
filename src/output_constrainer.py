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
    model: Any
    functions: List[FuncDefinition]

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
        pass
