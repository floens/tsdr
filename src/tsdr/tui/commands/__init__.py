"""Command framework for TSDR.

Importing this package triggers command registration via registry.py.
"""

from tsdr.tui.commands.base import Command, Completion
from tsdr.tui.commands.registry import (
    COMMANDS,
    execute,
    get_command_info,
    get_filtered_commands,
)

__all__ = [
    "COMMANDS",
    "Command",
    "Completion",
    "execute",
    "get_command_info",
    "get_filtered_commands",
]
