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
_MULTIPLEXER_VARS = ("TMUX", "STY", "ZELLIJ")
# Cell size in pixels when the terminal doesn't report it.
_DEFAULT_CELL = (10.0, 20.0)
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
    # Multiplexers swallow graphics escape sequences unless passthrough is configured, and they keep (or forward
    # over SSH) the outer terminal's KITTY_WINDOW_ID / LC_TERMINAL, so check them first.
    if any(env.get(var) for var in _MULTIPLEXER_VARS) or term.startswith(("screen", "tmux")):
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


class ImageDecodeError(ValueError):
    """The TAN image could not be decoded; no display mode can show it."""


def kitty_sequence(png: bytes, columns: int | None = None) -> str:
    """Transmit and display a PNG, split into <=4096 byte base64 chunks as the protocol requires.

    ``columns`` scales the image to that many cells wide (keeping the aspect ratio).
    """
    payload = base64.standard_b64encode(png).decode("ascii")
    chunks = [payload[i : i + _KITTY_CHUNK] for i in range(0, len(payload), _KITTY_CHUNK)] or [""]
    out = []
    for index, chunk in enumerate(chunks):
        more = 0 if index == len(chunks) - 1 else 1
        size = f",c={columns}" if columns else ""
        control = f"a=T,f=100{size},m={more}" if index == 0 else f"m={more}"
        out.append(f"\x1b_G{control};{chunk}\x1b\\")
    return "".join(out) + "\n"


def iterm_sequence(png: bytes, columns: int | None = None) -> str:
    payload = base64.standard_b64encode(png).decode("ascii")
    width = f";width={columns}" if columns else ""
    return f"\x1b]1337;File=inline=1;size={len(png)}{width};preserveAspectRatio=1:{payload}\x07\n"


def has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def flatten(image: Image.Image) -> Image.Image:
    """RGB on a white background, so transparent pixels don't turn the quiet zone black."""
    if has_alpha(image):
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return image.convert("RGB")


def as_png(data: bytes, image: Image.Image) -> bytes:
    """The original bytes if they are an opaque PNG already, otherwise a re-encode of the flattened RGB image.

    Transparent pixels would show the terminal background, which on a dark theme hides the quiet zone.
    """
    if image.format == "PNG" and not has_alpha(image):
        return data
    buffer = BytesIO()
    flatten(image).save(buffer, format="PNG")
    return buffer.getvalue()


def block_scale(image: Image.Image, columns: int, rows: int | None = None) -> float:
    scale = columns / image.width
    if rows is not None:
        scale = min(scale, 2 * rows / image.height)
    return scale


def module_size(image: Image.Image) -> int:
    """Estimate the module size in pixels: the shortest run of light or dark pixels inside sampled rows/columns.

    Every QR/matrix code has single-module runs; runs touching the image edge (quiet zone) are ignored.
    """
    gray = flatten(image).convert("L")
    width, height = gray.size
    pixels = gray.load()
    shortest = max(width, height)
    for horizontal in (True, False):
        lines, length = (height, width) if horizontal else (width, height)
        for line in range(0, lines, max(1, lines // 64)):
            runs, run, previous = [], 0, None
            for pos in range(length):
                dark = (pixels[pos, line] if horizontal else pixels[line, pos]) < 128
                if dark == previous:
                    run += 1
                else:
                    runs.append(run)
                    run, previous = 1, dark
            # runs[0] is the empty start, runs[1] touches the edge; the final run is never appended.
            shortest = min([shortest, *runs[2:]])
    return max(1, shortest)


def block_art(image: Image.Image, columns: int, rows: int | None = None) -> str:
    """Render with '▀' (upper half block): foreground = upper pixel, background = lower pixel.

    The image is shrunk to at most ``columns`` cells wide and ``rows`` text rows high (two pixels per row).
    """
    image = flatten(image)
    scale = block_scale(image, columns, rows)
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


def cell_pixels(out) -> tuple[float, float]:
    """Width and height of a terminal cell in pixels, as reported by the tty (sshd forwards it), or a guess."""
    try:
        import fcntl
        import struct
        import termios

        rows, columns, x_pixels, y_pixels = struct.unpack(
            "HHHH", fcntl.ioctl(out.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        )
        if rows and columns and x_pixels and y_pixels:
            return x_pixels / columns, y_pixels / rows
    except Exception:  # not a tty, no termios (Windows), ...
        pass
    return _DEFAULT_CELL


def fit_columns(image: Image.Image, out) -> int | None:
    """Cells to scale a graphics-protocol image down to so it fits the terminal, or None if it fits already."""
    size = shutil.get_terminal_size((80, 24))
    max_columns, max_rows = max(1, size.columns - 1), max(1, size.lines - 2)
    cell_width, cell_height = cell_pixels(out)
    columns, rows = image.width / cell_width, image.height / cell_height
    if columns <= max_columns and rows <= max_rows:
        return None
    return max(1, math.floor(columns * min(max_columns / columns, max_rows / rows)))


def warn_if_too_small(image: Image.Image, scale: float) -> None:
    """NEAREST shrinking below one pixel per module drops whole modules, and the code can't be decoded."""
    if scale >= 1:
        return
    module = module_size(image)
    if module * scale < 1:
        columns = math.ceil(image.width / module) + 1
        rows = math.ceil(image.height / module / 2) + 2
        logger.warning(
            "The terminal is too small to draw the TAN image scannably; enlarge it to at least %d columns x %d rows "
            "(or zoom out) and use cleanab.tan_display: window if that isn't possible",
            columns,
            rows,
        )


def decode(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise ImageDecodeError(f"Could not decode the TAN image: {e}") from e
    return image


def show_image(data: bytes, mode: TanDisplayMode, out=None) -> TanDisplayMode:
    """Display the image in a concrete mode (see ``resolve_mode``) and return it.

    Raises ``ImageDecodeError`` if the image itself is broken, anything else is an output problem.
    """
    if mode not in MODES or mode == "auto":
        raise ValueError(f"Unknown TAN display mode {mode!r}, expected one of {', '.join(MODES[1:])}")
    out = out or sys.stdout

    image = decode(data)
    if mode == "window":
        image.show()
        time.sleep(_WINDOW_DELAY)
        return mode
    if mode in ("kitty", "iterm"):
        build = kitty_sequence if mode == "kitty" else iterm_sequence
        sequence = build(as_png(data, image), fit_columns(image, out))
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
        columns, rows = max(1, min(size.columns - 1, _MAX_COLUMNS)), max(1, size.lines - 2)
        warn_if_too_small(image, block_scale(image, columns, rows))
        out.write(block_art(image, columns, rows))
    out.flush()
    return mode
