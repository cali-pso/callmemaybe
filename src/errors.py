class ParsingError(Exception):
    """Raise when JSON parsing or validation fails."""

    pass


class UnknownFunctionError(Exception):
    """Raise when the LLM outputs an unrecognised function name."""

    pass
