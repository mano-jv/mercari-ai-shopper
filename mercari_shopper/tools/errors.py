"""Errors a tool handler raises. The registry turns them into tool output."""


class ToolError(Exception):
    """Base error for a tool call that did not run cleanly."""


class ToolValidationError(ToolError):
    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"{tool_name}: {detail}")


class UnknownToolError(ToolError):
    def __init__(self, tool_name: str, known: tuple[str, ...]) -> None:
        self.tool_name = tool_name
        self.known = known
        joined = ", ".join(known) if known else "(none)"
        super().__init__(f"unknown tool {tool_name!r}; known tools: {joined}")
