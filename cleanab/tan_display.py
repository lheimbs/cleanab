"""Show the photoTAN / QR image of a TAN challenge where the user can actually see it.

Over SSH there is no desktop to open an image viewer on, so we draw into the terminal itself:

- ``kitty``: the kitty graphics protocol (kitty, Ghostty, Konsole, ...)
- ``iterm``: iTerm2 inline images (iTerm2, WezTerm, mintty)
- ``blocks``: truecolor half-block characters, which work in nearly every modern terminal
- ``window``: open the image with the local image viewer (PIL); only used when you ask for it

``auto`` (the default) picks a terminal mode from the environment when the first image is shown, never ``window``.
Set ``cleanab.tan_display`` in the config or the
``CLEANAB_TAN_DISPLAY`` environment variable to force a mode.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import math
import os
import re
import shutil
import sys
import time
from io import BytesIO
from typing import Literal, get_args

from logzero import logger
from PIL import Image, ImageShow

TanDisplayMode = Literal["auto", "kitty", "iterm", "blocks", "window"]
MODES: tuple[str, ...] = get_args(TanDisplayMode)
ENV_VAR = "CLEANAB_TAN_DISPLAY"
_KITTY_CHUNK = 4096
_MULTIPLEXER_VARS = ("TMUX", "STY", "ZELLIJ")
# Cell size in pixels when the terminal doesn't report it.
_DEFAULT_CELL = (10.0, 20.0)
# String terminators of the kitty APC (ESC _ G ... ESC \\) and iTerm2 OSC (ESC ] 1337 ... BEL) sequences.
_TERMINATORS = {"kitty": "\x1b\\\n", "iterm": "\x07\n"}
# A 1x1 RGB query image (kitty docs, "Querying support"), followed by a primary device attributes request (DA1):
# every terminal answers DA1, and one that speaks the graphics protocol answers the query before it.
_KITTY_QUERY = b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c"
_KITTY_OK = b"\x1b_Gi=31;OK"
_DA1_REPLY = re.compile(rb"\x1b\[\?[\d;]*c")
# Terminal replies that can end up in a typed line: terminated APC/OSC/DCS strings (kitty graphics replies) and CSI
# replies such as DA1. An unterminated string has no reliable end, so read_line asks again instead of guessing.
_TERMINAL_REPLY = re.compile(r"\x1b[_\]P][^\x07\x1b]*(?:\x1b\\|\x07)|\x1b\[[\d;?]*[A-Za-z]")
_PROBE_TIMEOUT = 2.0
# Give whatever application PIL uses to display the image time to start.
_WINDOW_DELAY = 5


def configured_mode(configured: TanDisplayMode = "auto", env: dict[str, str] | None = None) -> TanDisplayMode:
    """Apply the CLEANAB_TAN_DISPLAY override, at startup; the result may still be ``auto``.

    An invalid override is ignored with a warning instead of failing later.
    """
    env = os.environ if env is None else env
    override = env.get(ENV_VAR, "").strip().lower()
    if override in MODES:
        return override  # type: ignore[return-value]
    if override:
        logger.warning(
            "Ignoring %s=%r, expected one of %s; using %r", ENV_VAR, env[ENV_VAR], ", ".join(MODES), configured
        )
    return configured


@functools.cache
def resolve_once(mode: TanDisplayMode) -> TanDisplayMode:
    """Resolve ``auto`` for this terminal on the first TAN image, then remember it.

    Detection may ask the terminal (up to two seconds), so runs that never see a TAN image don't pay for it.
    """
    return mode if mode != "auto" else _auto_mode(os.environ)


def _auto_mode(env) -> TanDisplayMode:
    mode = detect_mode(env)
    # KITTY_WINDOW_ID / KONSOLE_VERSION leak into terminals started from kitty or Konsole (VS Code, xterm, su, ...),
    # and those silently drop the graphics sequence, so the fallback to blocks never happens. Ask the terminal.
    if mode == "kitty" and kitty_graphics_supported() is False:
        mode = detect_mode(env, kitty=False)
        logger.info("The terminal doesn't support kitty graphics, drawing TAN images with tan_display=%s", mode)
    if mode == "blocks" and (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        logger.info(
            "TAN images will be drawn with text blocks; if your bank app can't scan them, set "
            "cleanab.tan_display: window to open them in the image viewer instead"
        )
    return mode


def detect_mode(env: dict[str, str] | None = None, *, kitty: bool = True) -> TanDisplayMode:
    env = os.environ if env is None else env
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    # Multiplexers swallow graphics escape sequences unless passthrough is configured, and they keep (or forward
    # over SSH) the outer terminal's KITTY_WINDOW_ID / LC_TERMINAL, so check them first.
    if any(env.get(var) for var in _MULTIPLEXER_VARS) or term.startswith(("screen", "tmux")):
        return "blocks"
    if kitty and (
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


def kitty_graphics_supported(timeout: float = _PROBE_TIMEOUT) -> bool | None:
    """Ask the controlling terminal whether it speaks the kitty graphics protocol; None if we can't tell."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import select
        import termios
        import tty
    except ImportError:  # Windows
        return None
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return None
    try:
        saved = termios.tcgetattr(fd)
        try:
            # No echo and no line buffering, so the reply neither shows up on screen nor waits for Enter.
            tty.setcbreak(fd)
            os.write(fd, _KITTY_QUERY)
            # A reply arriving after we give up is stripped from the next prompt by read_line.
            reply, deadline = b"", time.monotonic() + timeout
            while not _DA1_REPLY.search(reply):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                    return None
                chunk = os.read(fd, 1024)
                if not chunk:
                    return None
                reply += chunk
            return _KITTY_OK in reply
        finally:
            # TCSAFLUSH drops any unread reply, so it can't end up in the TAN prompt.
            termios.tcsetattr(fd, termios.TCSAFLUSH, saved)
    except (OSError, termios.error):
        return None
    finally:
        os.close(fd)


class ImageDecodeError(ValueError):
    """The TAN image could not be decoded; no display mode can show it."""


def kitty_sequence(png: bytes, columns: int | None = None) -> str:
    """Transmit and display a PNG, split into <=4096 byte base64 chunks as the protocol requires.

    ``q=2`` keeps the terminal from replying: the reply would land in stdin and be read as part of the TAN.

    ``columns`` scales the image to that many cells wide (keeping the aspect ratio).
    """
    payload = base64.standard_b64encode(png).decode("ascii")
    chunks = [payload[i : i + _KITTY_CHUNK] for i in range(0, len(payload), _KITTY_CHUNK)] or [""]
    out = []
    for index, chunk in enumerate(chunks):
        more = 0 if index == len(chunks) - 1 else 1
        size = f",c={columns}" if columns else ""
        control = f"a=T,f=100{size},q=2,m={more}" if index == 0 else f"q=2,m={more}"
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
    return image if image.mode == "RGB" else image.convert("RGB")


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


# Sampled rows/columns per direction, and the share of the shortest runs treated as noise (anti-aliasing, JPEG).
_MODULE_SAMPLES = 32
_MODULE_PERCENTILE = 0.1


def module_size(image: Image.Image) -> int:
    """Estimate the module size in pixels from the runs of light or dark pixels in sampled rows and columns.

    Single-module runs are the most common runs in a QR/matrix code, so a low percentile finds them while a few
    one-pixel runs from noise don't drag the estimate down. Runs touching the image edge (quiet zone) are ignored.
    Expects a flattened (RGB) image.
    """
    dark = image.convert("L").point(lambda value: 255 if value < 128 else 0)
    width, height = dark.size
    data = dark.tobytes()
    runs: list[int] = []
    for horizontal in (True, False):
        lines, length = (height, width) if horizontal else (width, height)
        for line in range(0, lines, max(1, lines // _MODULE_SAMPLES)):
            if horizontal:
                values = data[line * width : (line + 1) * width]
            else:
                values = data[line::width]
            run, edges = 1, []
            for pos in range(1, length):
                if values[pos] == values[pos - 1]:
                    run += 1
                else:
                    edges.append(run)
                    run = 1
            # edges[0] touches the start of the line; the final run (touching the end) is never appended.
            runs.extend(edges[1:])
    if not runs:
        return max(width, height)
    runs.sort()
    return max(1, runs[int(len(runs) * _MODULE_PERCENTILE)])


def color_depth(env: dict[str, str] | None = None) -> Literal["truecolor", "256", "16"]:
    """How many colors the terminal can show, as far as the environment tells."""
    env = os.environ if env is None else env
    term = env.get("TERM", "")
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit") or any(
        name in term for name in ("kitty", "ghostty", "wezterm", "alacritty", "foot", "direct")
    ):
        return "truecolor"
    # sshd doesn't forward COLORTERM, so 256 colors is the safe choice for xterm-256color and friends.
    if "256" in term:
        return "256"
    return "16"


_CUBE = (0, 95, 135, 175, 215, 255)
# The 16 standard colors as xterm draws them; the SGR code is 30-37 / 90-97 (foreground), +10 for background.
_BASIC = [
    ((0, 0, 0), 30), ((205, 0, 0), 31), ((0, 205, 0), 32), ((205, 205, 0), 33),
    ((0, 0, 238), 34), ((205, 0, 205), 35), ((0, 205, 205), 36), ((229, 229, 229), 37),
    ((127, 127, 127), 90), ((255, 0, 0), 91), ((0, 255, 0), 92), ((255, 255, 0), 93),
    ((92, 92, 255), 94), ((255, 0, 255), 95), ((0, 255, 255), 96), ((255, 255, 255), 97),
]  # fmt: skip


def _distance(a, b) -> int:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _ansi256(rgb) -> int:
    """The nearest xterm-256 color: from the 6x6x6 cube or the gray ramp. Exact for black and white."""
    cube = [min(range(6), key=lambda i: abs(_CUBE[i] - value)) for value in rgb]
    cube_rgb = tuple(_CUBE[i] for i in cube)
    gray = min(23, max(0, round((sum(rgb) / 3 - 8) / 10)))
    gray_rgb = (8 + 10 * gray,) * 3
    if _distance(rgb, gray_rgb) < _distance(rgb, cube_rgb):
        return 232 + gray
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]


def sgr_color(rgb, background: bool, depth: str = "truecolor") -> str:
    if depth == "truecolor":
        return "\x1b[{};2;{};{};{}m".format(48 if background else 38, *rgb)
    if depth == "256":
        return f"\x1b[{48 if background else 38};5;{_ansi256(rgb)}m"
    code = min(_BASIC, key=lambda entry: _distance(rgb, entry[0]))[1]
    return f"\x1b[{code + 10 if background else code}m"


def block_art(image: Image.Image, columns: int, rows: int | None = None, depth: str = "truecolor") -> str:
    """Render with '▀' (upper half block): foreground = upper pixel, background = lower pixel.

    The image is shrunk to at most ``columns`` cells wide and ``rows`` text rows high (two pixels per row).
    ``depth`` is the terminal's color depth, see ``color_depth``.
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
                cells.append(sgr_color(top, background=False, depth=depth))
                fg = top
            if bottom != bg:
                cells.append(sgr_color(bottom, background=True, depth=depth))
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


def terminal_box(out) -> tuple[int, int]:
    """Columns and rows an image may use: one column less against wrapping, two rows less for the TAN prompt."""
    try:
        size = os.get_terminal_size(out.fileno())
    except (AttributeError, OSError, ValueError):  # not a tty, or no file descriptor (io.UnsupportedOperation)
        size = shutil.get_terminal_size((80, 24))
    return max(1, size.columns - 1), max(1, size.lines - 2)


def fit_columns(image: Image.Image, out) -> int | None:
    """Cells to scale a graphics-protocol image down to so it fits the terminal, or None if it fits already."""
    max_columns, max_rows = terminal_box(out)
    cell_width, cell_height = cell_pixels(out)
    columns, rows = image.width / cell_width, image.height / cell_height
    if columns <= max_columns and rows <= max_rows:
        return None
    return max(1, math.floor(columns * min(max_columns / columns, max_rows / rows)))


def warn_if_too_small(image: Image.Image, scale: float) -> None:
    """NEAREST shrinking below one pixel per module drops whole modules, and the code can't be decoded.

    Expects a flattened (RGB) image.
    """
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


def show_window(image: Image.Image, env: dict[str, str] | None = None) -> None:
    """Open the image viewer; raise ``OSError`` when it clearly can't work, so the caller can fall back.

    Viewers start in the background, so a viewer that fails later (e.g. xdg-open) goes unnoticed.
    """
    env = os.environ if env is None else env
    if sys.platform not in ("darwin", "win32") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise OSError("no graphical display (DISPLAY/WAYLAND_DISPLAY) to open the image viewer on")
    if not ImageShow.show(image):
        raise OSError("no image viewer found")
    time.sleep(_WINDOW_DELAY)


def read_line(prompt: str) -> str:
    """``input()`` without terminal replies: a late kitty probe or graphics reply would otherwise become the input.

    Replies already waiting are discarded before the prompt; ones that arrive while typing are stripped. If what
    is left still contains an escape character (a reply cut in half, a stray ESC key), the line can't be trusted and
    is asked for again, rather than sending a mangled TAN to the bank.
    """
    while True:
        if sys.stdin.isatty():
            with contextlib.suppress(Exception):
                import termios

                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        line = _TERMINAL_REPLY.sub("", input(prompt))
        if "\x1b" not in line:
            return line
        logger.error("The input was mixed up with a reply from the terminal, please enter it again")


def decode(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise ImageDecodeError(f"Could not decode the TAN image: {e}") from e
    return image


def show_image(data: bytes, mode: TanDisplayMode, out=None, *, image: Image.Image | None = None) -> TanDisplayMode:
    """Display the image in a concrete mode (see ``resolve_once``) and return it.

    ``image`` is ``data`` decoded already, so falling back to another mode needn't decode it again.
    Raises ``ImageDecodeError`` if the image itself is broken, anything else is an output problem.
    """
    if mode not in MODES or mode == "auto":
        raise ValueError(f"Unknown TAN display mode {mode!r}, expected one of {', '.join(MODES[1:])}")
    out = out or sys.stdout

    image = decode(data) if image is None else image
    if mode == "window":
        show_window(image)
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
        columns, rows = terminal_box(out)
        image = flatten(image)
        warn_if_too_small(image, block_scale(image, columns, rows))
        out.write(block_art(image, columns, rows, color_depth()))
    out.flush()
    return mode
