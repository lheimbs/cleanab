"""Show the photoTAN / QR image of a TAN challenge where the user can actually see it.

Over SSH there is no desktop to open an image viewer on, so we draw into the terminal itself:

- ``kitty``: the kitty graphics protocol (kitty, Ghostty, WezTerm, Konsole, ...)
- ``iterm``: iTerm2 inline images (iTerm2, WezTerm, mintty)
- ``blocks``: truecolor half-block characters, which work in nearly every modern terminal
- ``window``: open the image with the local image viewer (PIL); only used when you ask for it

``auto`` (the default) picks a terminal mode from the environment and never ``window``.
Set ``cleanab.tan_display`` in the config or the
``CLEANAB_TAN_DISPLAY`` environment variable to force a mode.
"""

from __future__ import annotations

import base64
import os
import shutil
import sys
from io import BytesIO

from PIL import Image

MODES = ("auto", "kitty", "iterm", "blocks", "window")
_KITTY_CHUNK = 4096

_mode = "auto"


def set_mode(mode: str) -> None:
    global _mode
    _mode = mode


def detect_mode(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    # Inside tmux/screen, graphics escape sequences are swallowed unless passthrough is configured.
    if env.get("TMUX") or term.startswith("screen"):
        return "blocks"
    if env.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term or program in ("ghostty", "WezTerm"):
        return "kitty"
    if program == "iTerm.app" or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    # Never guess "window": multiplexers and session managers (tmux, herdr, ...) drop SSH_* variables while the
    # machine still has a desktop, so the viewer would open on the remote screen. Opt in with tan_display: window.
    return "blocks"


def kitty_sequence(png: bytes) -> str:
    """Transmit and display a PNG, split into <=4096 byte base64 chunks as the protocol requires."""
    payload = base64.standard_b64encode(png).decode("ascii")
    chunks = [payload[i : i + _KITTY_CHUNK] for i in range(0, len(payload), _KITTY_CHUNK)] or [""]
    out = []
    for index, chunk in enumerate(chunks):
        more = 0 if index == len(chunks) - 1 else 1
        control = f"a=T,f=100,m={more}" if index == 0 else f"m={more}"
        out.append(f"\x1b_G{control};{chunk}\x1b\\")
    return "".join(out) + "\n"


def iterm_sequence(png: bytes) -> str:
    payload = base64.standard_b64encode(png).decode("ascii")
    return f"\x1b]1337;File=inline=1;size={len(png)};preserveAspectRatio=1:{payload}\x07\n"


def block_art(image: Image.Image, columns: int) -> str:
    """Render with '▀' (upper half block): foreground = upper pixel, background = lower pixel."""
    image = image.convert("RGB")
    if image.width > columns:
        height = max(2, round(image.height * columns / image.width))
        image = image.resize((columns, height), Image.BOX)
    pixels = image.load()
    lines = []
    for y in range(0, image.height, 2):
        cells = []
        for x in range(image.width):
            top = pixels[x, y]
            bottom = pixels[x, y + 1] if y + 1 < image.height else (255, 255, 255)
            cells.append("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀".format(*top, *bottom))
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines) + "\n"


def show_image(data: bytes, mode: str | None = None, out=None) -> str:
    """Display the image and return the mode that was used."""
    mode = mode or os.environ.get("CLEANAB_TAN_DISPLAY") or _mode
    if mode not in MODES:
        raise ValueError(f"Unknown TAN display mode {mode!r}, expected one of {', '.join(MODES)}")
    if mode == "auto":
        mode = detect_mode()
    out = out or sys.stdout

    image = Image.open(BytesIO(data))
    if mode == "window":
        image.show()
        return mode
    if mode in ("kitty", "iterm"):
        png = BytesIO()
        image.convert("RGBA" if image.mode in ("P", "LA") else image.mode).save(png, format="PNG")
        out.write((kitty_sequence if mode == "kitty" else iterm_sequence)(png.getvalue()))
    else:
        columns = max(20, min(shutil.get_terminal_size((80, 24)).columns - 1, 100))
        out.write(block_art(image, columns))
    out.flush()
    return mode
