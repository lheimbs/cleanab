import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from cleanab import fints as fints_module
from cleanab import tan_display


def png_bytes(size=(8, 8), color=(0, 0, 0), mode="RGB", fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"TERM": "xterm-ghostty"}, "kitty"),
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
        ({"TERM_PROGRAM": "WezTerm"}, "iterm"),
        ({"TERM_PROGRAM": "mintty"}, "iterm"),
        ({"TERM": "xterm-256color", "KONSOLE_VERSION": "240202"}, "kitty"),
        ({"TERM": "xterm-kitty", "TMUX": "/tmp/tmux"}, "blocks"),
        ({"TERM": "screen-256color"}, "blocks"),
        ({"SSH_CONNECTION": "a b c d", "DISPLAY": ":0"}, "blocks"),
        ({"DISPLAY": ":0"}, "blocks"),
        ({"WAYLAND_DISPLAY": "wayland-0"}, "blocks"),
        ({}, "blocks"),
    ],
)
def test_detect_mode(env, expected):
    assert tan_display.detect_mode(env) == expected


def test_modes_match_literal():
    assert tan_display.MODES == ("auto", "kitty", "iterm", "blocks", "window")


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "window"),
        ({"CLEANAB_TAN_DISPLAY": "iterm"}, "iterm"),
        ({"CLEANAB_TAN_DISPLAY": " Blocks "}, "blocks"),
        ({"CLEANAB_TAN_DISPLAY": "sixel"}, "window"),
        ({"CLEANAB_TAN_DISPLAY": "auto", "TERM": "xterm-kitty"}, "kitty"),
    ],
)
def test_resolve_mode(env, expected):
    assert tan_display.resolve_mode("window", env) == expected


def test_resolve_mode_never_returns_auto():
    assert tan_display.resolve_mode("auto", {"TERM_PROGRAM": "WezTerm"}) == "iterm"
    assert tan_display.resolve_mode("auto", {}) == "blocks"


@pytest.mark.parametrize(("env", "hint"), [({"DISPLAY": ":0"}, True), ({"WAYLAND_DISPLAY": "w"}, True), ({}, False)])
def test_resolve_mode_hints_window_on_desktop(env, hint, monkeypatch):
    messages = []
    monkeypatch.setattr(tan_display.logger, "info", lambda msg, *a: messages.append(msg))
    assert tan_display.resolve_mode("auto", env) == "blocks"
    assert any("tan_display: window" in m for m in messages) == hint


def test_kitty_sequence_chunks_and_flags():
    data = bytes(range(256)) * 40  # 10240 bytes -> 13654 base64 chars -> 4 chunks
    sequence = tan_display.kitty_sequence(data)
    chunks = [c for c in sequence.split("\x1b\\") if c.strip()]
    assert len(chunks) == 4
    assert chunks[0].startswith("\x1b_Ga=T,f=100,m=1;")
    assert chunks[1].startswith("\x1b_Gm=1;")
    assert chunks[-1].startswith("\x1b_Gm=0;")
    payload = "".join(c.split(";", 1)[1] for c in chunks)
    assert base64.b64decode(payload) == data
    assert all(len(c.split(";", 1)[1]) <= 4096 for c in chunks)


def test_iterm_sequence():
    sequence = tan_display.iterm_sequence(b"abc")
    assert sequence.startswith("\x1b]1337;File=inline=1;size=3;")
    assert base64.b64encode(b"abc").decode() in sequence


def test_block_art_pairs_rows_and_scales():
    image = Image.new("RGB", (4, 4), (255, 255, 255))
    image.putpixel((0, 0), (0, 0, 0))
    art = tan_display.block_art(image, columns=40).splitlines()
    assert len(art) == 2  # 4 pixel rows -> 2 text rows
    assert art[0].startswith("\x1b[38;2;0;0;0m\x1b[48;2;255;255;255m▀\x1b[38;2;255;255;255m▀▀▀")
    scaled = tan_display.block_art(Image.new("RGB", (100, 100)), columns=20).splitlines()
    assert len(scaled) == 10


def test_block_art_fits_terminal_height():
    # 400x400 on a 100x30 terminal: width alone would allow 99 columns / 50 rows
    art = tan_display.block_art(Image.new("RGB", (400, 400)), columns=99, rows=28).splitlines()
    assert len(art) <= 28
    assert art[0].count("▀") <= 56


def test_block_art_narrow_pane_does_not_overflow():
    art = tan_display.block_art(Image.new("RGB", (100, 100)), columns=10).splitlines()
    assert all(line.count("▀") <= 10 for line in art)


def test_block_art_keeps_qr_modules_crisp():
    # 3-pixel black/white stripes shrunk by a non-integer factor must not produce gray
    image = Image.new("RGB", (99, 4), "white")
    for x in range(99):
        if (x // 3) % 2 == 0:
            for y in range(4):
                image.putpixel((x, y), (0, 0, 0))
    art = tan_display.block_art(image, columns=40)
    colors = set(art.replace("\x1b[0m", "").replace("▀", "").replace("\n", "").split("\x1b["))
    assert colors <= {"", "38;2;0;0;0m", "38;2;255;255;255m", "48;2;0;0;0m", "48;2;255;255;255m"}


def test_block_art_only_emits_color_changes():
    art = tan_display.block_art(Image.new("RGB", (50, 2), "white"), columns=50)
    assert art == "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m" + "▀" * 50 + "\x1b[0m\n"


def test_block_art_transparent_background_is_white():
    art = tan_display.block_art(Image.new("RGBA", (2, 2), (0, 0, 0, 0)), columns=10)
    assert "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m▀" in art


def test_show_image_writes_selected_mode():
    out = io.StringIO()
    assert tan_display.show_image(png_bytes(), mode="kitty", out=out) == "kitty"
    assert out.getvalue().startswith("\x1b_Ga=T")
    out = io.StringIO()
    assert tan_display.show_image(png_bytes(), mode="blocks", out=out) == "blocks"
    assert "▀" in out.getvalue()


def test_show_image_passes_png_through_unchanged():
    data = png_bytes()
    out = io.StringIO()
    tan_display.show_image(data, mode="iterm", out=out)
    assert base64.b64encode(data).decode() in out.getvalue()


@pytest.mark.parametrize("mode", ["CMYK", "RGBA"])
def test_show_image_reencodes_other_formats(mode):
    color = (0, 0, 0, 0) if mode == "CMYK" else (0, 0, 0, 255)
    data = png_bytes(mode=mode, color=color, fmt="TIFF")
    out = io.StringIO()
    tan_display.show_image(data, mode="iterm", out=out)
    payload = out.getvalue().split(":", 1)[1].rstrip("\x07\n")
    assert Image.open(io.BytesIO(base64.b64decode(payload))).format == "PNG"


class BrokenOut(io.StringIO):
    def write(self, text):
        if text.startswith("\x1b_G") or text.startswith("\x1b]1337"):
            super().write(text[:10])
            raise OSError("broken pipe")
        return super().write(text)


@pytest.mark.parametrize(("mode", "terminator"), [("kitty", "\x1b\\"), ("iterm", "\x07")])
def test_show_image_terminates_escape_after_failed_write(mode, terminator):
    out = BrokenOut()
    with pytest.raises(OSError):
        tan_display.show_image(png_bytes(), mode=mode, out=out)
    assert out.getvalue().endswith(terminator + "\n")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unknown TAN display mode"):
        tan_display.show_image(png_bytes(), mode="sixel", out=io.StringIO())


def matrix_response():
    return SimpleNamespace(
        challenge="scan", challenge_hhduc=None, challenge_matrix=("image/png", b"PNGDATA"), challenge_html=""
    )


def test_handle_tan_response_shows_matrix_image(monkeypatch):
    shown = []
    monkeypatch.setattr(fints_module, "show_image", lambda data, mode: shown.append((data, mode)))
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    client = SimpleNamespace(send_tan=lambda r, tan: ["tx"], tan_display="kitty")
    assert fints_module.handle_tan_response(client, matrix_response()) == ["tx"]
    assert shown == [(b"PNGDATA", "kitty")]


def test_handle_tan_response_survives_display_failure(monkeypatch):
    attempts = []

    def broken(data, mode):
        attempts.append(mode)
        raise OSError("cannot write mode CMYK as PNG")

    monkeypatch.setattr(fints_module, "show_image", broken)
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    client = SimpleNamespace(send_tan=lambda r, tan: ["tx"], tan_display="kitty")
    assert fints_module.handle_tan_response(client, matrix_response()) == ["tx"]
    assert attempts == ["kitty", "blocks"]


def test_show_tan_image_resolves_auto_before_falling_back(monkeypatch):
    attempts = []

    def broken(data, mode):
        attempts.append(mode)
        raise OSError("cannot identify image file")

    monkeypatch.setattr(fints_module, "show_image", broken)
    monkeypatch.setattr(fints_module, "detect_mode", lambda: "blocks")
    fints_module.show_tan_image(b"junk", "auto")
    assert attempts == ["blocks"]
