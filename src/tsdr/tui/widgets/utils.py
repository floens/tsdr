from textual.widgets import OptionList


class NonFocusableOptionList(OptionList):
    """OptionList that only responds to mouse clicks, not keyboard focus."""

    can_focus = False
