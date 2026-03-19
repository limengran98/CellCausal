from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseTool(ABC):
    """Base class for low-level V2 tools."""

    name: ClassVar[str] = "base-tool"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with tool-specific arguments."""
