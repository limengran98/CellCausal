from __future__ import annotations

from typing import Dict

from ..tools.base import BaseTool


class ToolRegistry:
    """Minimal registry for repo-native V2 tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, name: str, tool: BaseTool) -> None:
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]
