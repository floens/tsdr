from argparse import Namespace

from tsdr.core import storage
from tsdr.tui.commands.base import Command, CommandParser


class PathsCommand(Command):
    @property
    def description(self) -> str:
        return "Show the user config directory and its contents"

    def configure(self, parser: CommandParser) -> None:
        pass

    def run(self, args: Namespace) -> str:
        cfg = storage.config_dir()
        lines = [f"Config directory: {cfg}"]
        if not cfg.exists():
            lines.append("  (does not exist yet)")
            return "\n".join(lines)

        try:
            entries = sorted(cfg.iterdir())
        except OSError as e:
            lines.append(f"  (error listing: {e})")
            return "\n".join(lines)

        if not entries:
            lines.append("  (empty)")
            return "\n".join(lines)

        for entry in entries:
            if entry.is_dir():
                try:
                    count = sum(1 for _ in entry.iterdir())
                    lines.append(f"  {entry.name}/ ({count} items)")
                except OSError:
                    lines.append(f"  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                    lines.append(f"  {entry.name} ({size:,} bytes)")
                except OSError:
                    lines.append(f"  {entry.name}")
        return "\n".join(lines)
