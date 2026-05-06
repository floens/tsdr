## TSDR - The scriptable SDR for the terminal

TSDR is a terminal user interface for software defined radio.

It gives you a waterfall, spectrum display, and a bunch of demodulators. It is also powerful and command line focused.

! TSDR is in heavy development.

### Installation

The project is not on pipy, you can install directly from GitHub instead:

```
uv tool install --from git+https://github.com/floens/tsdr tsdr
```

Ghostty is recommended for its excellent feature set and flicker-free image support. For Windows, try Kitty or WezTerm.

### Goals

_See also [DESIGN.md](DESIGN.md)_

Terminal based interface, but using image support where needed. This should make it great to be scriptiple, so TSDR can be used as generic SDR toolkit, as well as a powerful tool to discover sdr.

Use of Python for a low barrier to read and modify the code. Use of numba/numpy for performance, where necessary.
