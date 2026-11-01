"""Show the photoTAN / QR image of a TAN challenge where the user can actually see it.

Over SSH there is no desktop to open an image viewer on, so we draw into the terminal itself:

- ``kitty``: the kitty graphics protocol (kitty, Ghostty, Konsole, ...)
- ``iterm``: iTerm2 inline images (iTerm2, WezTerm, mintty)
- ``blocks``: truecolor half-block characters, which work in nearly every modern terminal
- ``window``: open the image with the local image viewer (PIL); only used when you ask for it

``auto`` (the default) picks a terminal mode from the environment at startup and never ``window``.
Set ``cleanab.tan_display`` in the config or the
``CLEANAB_TAN_DISPLAY`` environment variable to force a mode.
"""

from __future__ import annotations

import base64
import contextlib
import math
import os
import shutil
import sys
import time
from io import BytesIO
from typing import Literal, get_args

from logzero import logger
from PIL import Image

TanDisplayMode = Literal["auto", "kitty", "iterm", "blocks", "window"]
MODES: tuple[str, ...] = get_args(TanDisplayMode)
ENV_VAR = "CLEANAB_TAN_DISPLAY"
_KITTY_CHUNK = 4096
_MAX_COLUMNS = 100
# String terminators of the kitty APC (ESC _ G ... ESC \\) and iTerm2 OSC (ESC ] 1337 ... BEL) sequences.
_TERMINATORS = {"kitty": "\x1b\\\n", "iterm": "\x07\n"}
# Give whatever application PIL uses to display the image time to start.
_WINDOW_DELAY = 5


def resolve_mode(configured: TanDisplayMode = "auto", env: dict[str, str] | None = None) -> TanDisplayMode:
    """Turn the configured mode into a concrete one (never ``auto``), once at startup.

    The CLEANAB_TAN_DISPLAY override wins; an invalid value is ignored with a warning instead of failing later.
    """
    env = os.environ if env is None else env
    mode = configured
    override = env.get(ENV_VAR, "").strip().lower()
    if override in MODES:
        mode = override  # type: ignore[assignment]
    elif override:
        logger.warning(
            "Ignoring %s=%r, expected one of %s; using %r", ENV_VAR, env[ENV_VAR], ", ".join(MODES), configured
        )
    if mode != "auto":
        return mode
    mode = detect_mode(env)
    if mode == "blocks" and (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        logger.info(
            "TAN images will be drawn with text blocks; if your bank app can't scan them, set "
            "cleanab.tan_display: window to open them in the image viewer instead"
        )
    return mode


def detect_mode(env: dict[str, str] | None = None) -> TanDisplayMode:
    env = os.environ if env is None else env
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    # Inside tmux/screen, graphics escape sequences are swallowed unless passthrough is configured.
    if env.get("TMUX") or term.startswith("screen"):
        return "blocks"
    if (
        env.get("KITTY_WINDOW_ID")
        or "kitty" in term
        or "ghostty" in term
        or program == "ghostty"
        # Konsole keeps TERM=xterm-256color but supports the kitty graphics protocol.
        or env.get("KONSOLE_VERSION")
    ):
        return "kitty"
    # WezTerm only speaks the kitty protocol with enable_kitty_graphics, but iTerm2 images work out of the box.
    if program in ("iTerm.app", "WezTerm", "mintty") or env.get("LC_TERMINAL") == "iTerm2":
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


def flatten(image: Image.Image) -> Image.Image:
    """RGB on a white background, so transparent pixels don't turn the quiet zone black."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return image.convert("RGB")


def as_png(data: bytes, image: Image.Image) -> bytes:
    """The original bytes if they are a PNG already, otherwise a re-encode of the flattened RGB image."""
    if image.format == "PNG":
        return data
    buffer = BytesIO()
    flatten(image).save(buffer, format="PNG")
    return buffer.getvalue()


def block_art(image: Image.Image, columns: int, rows: int | None = None) -> str:
    """Render with '▀' (upper half block): foreground = upper pixel, background = lower pixel.

    The image is shrunk to at most ``columns`` cells wide and ``rows`` text rows high (two pixels per row).
    """
    image = flatten(image)
    scale = columns / image.width
    if rows is not None:
        scale = min(scale, 2 * rows / image.height)
    if scale < 1:
        size = (max(1, math.floor(image.width * scale)), max(1, math.floor(image.height * scale)))
        # NEAREST keeps QR modules crisp; averaging blurs their edges into gray that some apps can't decode.
        image = image.resize(size, Image.NEAREST)
    pixels = image.load()
    lines = []
    for y in range(0, image.height, 2):
        cells = []
        fg = bg = None
        for x in range(image.width):
            top = pixels[x, y]
            bottom = pixels[x, y + 1] if y + 1 < image.height else (255, 255, 255)
            # Only emit colors that changed: QR codes are mostly runs, which keeps the output small over SSH.
            if top != fg:
                cells.append("\x1b[38;2;{};{};{}m".format(*top))
                fg = top
            if bottom != bg:
                cells.append("\x1b[48;2;{};{};{}m".format(*bottom))
                bg = bottom
            cells.append("▀")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines) + "\n"


def show_image(data: bytes, mode: TanDisplayMode = "auto", out=None) -> TanDisplayMode:
    """Display the image and return the mode that was used."""
    if mode not in MODES:
        raise ValueError(f"Unknown TAN display mode {mode!r}, expected one of {', '.join(MODES)}")
    if mode == "auto":
        mode = detect_mode()
    out = out or sys.stdout

    image = Image.open(BytesIO(data))
    if mode == "window":
        image.show()
        time.sleep(_WINDOW_DELAY)
        return mode
    if mode in ("kitty", "iterm"):
        sequence = (kitty_sequence if mode == "kitty" else iterm_sequence)(as_png(data, image))
        try:
            out.write(sequence)
        except Exception:
            # A partial write leaves the terminal inside the escape string, swallowing everything after it.
            with contextlib.suppress(Exception):
                out.write(_TERMINATORS[mode])
                out.flush()
            raise
    else:
        size = shutil.get_terminal_size((80, 24))
        # Leave one column against wrapping and two rows for the TAN prompt.
        out.write(block_art(image, max(1, min(size.columns - 1, _MAX_COLUMNS)), max(1, size.lines - 2)))
    out.flush()
    return mode
