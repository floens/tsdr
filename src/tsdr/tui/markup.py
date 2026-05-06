def escape_forced(text: str) -> str:
    """Escape Rich markup by replacing ``[`` with ``_``.

    `rich.markup.escape` mishandles some inputs containing unbalanced brackets,
    so we substitute instead of escaping.
    """
    return text.replace("[", "_")
