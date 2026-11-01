import base64
import io
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from cleanab import fints as fints_module
from cleanab import tan_display


def png_bytes(size=(8, 8), color=(0, 0, 0), mode="RGB", fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def no_terminal_probe(monkeypatch):
    """Never talk to the real terminal from tests; "can't tell" keeps the environment's guess."""
    monkeypatch.setattr(tan_display, "kitty_graphics_supported", lambda: None)
    tan_display.resolve_once.cache_clear()
    yield
    tan_display.resolve_once.cache_clear()


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
        ({"TERM": "tmux-256color", "LC_TERMINAL": "iTerm2"}, "blocks"),
        ({"KITTY_WINDOW_ID": "1", "ZELLIJ": "0"}, "blocks"),
        ({"KITTY_WINDOW_ID": "1", "STY": "1234.pts-0"}, "blocks"),
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
def test_configured_mode(env, expected):
    mode = tan_display.configured_mode("window", env)
    assert (mode if mode != "auto" else tan_display._auto_mode(env)) == expected


def test_auto_mode_never_returns_auto():
    assert tan_display._auto_mode({"TERM_PROGRAM": "WezTerm"}) == "iterm"
    assert tan_display._auto_mode({}) == "blocks"


@pytest.mark.parametrize(
    ("env", "supported", "expected"),
    [
        ({"KITTY_WINDOW_ID": "1"}, True, "kitty"),
        ({"KITTY_WINDOW_ID": "1"}, None, "kitty"),
        # Inherited from kitty/Konsole into a terminal without the graphics protocol.
        ({"KITTY_WINDOW_ID": "1", "TERM": "xterm"}, False, "blocks"),
        ({"KONSOLE_VERSION": "240202", "TERM_PROGRAM": "vscode"}, False, "blocks"),
        ({"KONSOLE_VERSION": "240202", "TERM_PROGRAM": "WezTerm"}, False, "iterm"),
    ],
)
def test_auto_mode_confirms_kitty_with_the_terminal(env, supported, expected, monkeypatch):
    monkeypatch.setattr(tan_display, "kitty_graphics_supported", lambda: supported)
    assert tan_display._auto_mode(env) == expected


def test_resolve_once_does_not_probe_explicit_kitty(monkeypatch):
    monkeypatch.setattr(tan_display, "kitty_graphics_supported", lambda: pytest.fail("probed"))
    assert tan_display.resolve_once("kitty") == "kitty"


def test_kitty_probe_needs_a_terminal(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(tan_display.sys, "stdin", io.StringIO())
    assert tan_display.kitty_graphics_supported() is None


@pytest.mark.parametrize(("env", "hint"), [({"DISPLAY": ":0"}, True), ({"WAYLAND_DISPLAY": "w"}, True), ({}, False)])
def test_auto_mode_hints_window_on_desktop(env, hint, monkeypatch):
    messages = []
    monkeypatch.setattr(tan_display.logger, "info", lambda msg, *a: messages.append(msg))
    assert tan_display._auto_mode(env) == "blocks"
    assert any("tan_display: window" in m for m in messages) == hint


def test_kitty_sequence_chunks_and_flags():
    data = bytes(range(256)) * 40  # 10240 bytes -> 13654 base64 chars -> 4 chunks
    sequence = tan_display.kitty_sequence(data)
    chunks = [c for c in sequence.split("\x1b\\") if c.strip()]
    assert len(chunks) == 4
    assert chunks[0].startswith("\x1b_Ga=T,f=100,q=2,m=1;")
    assert chunks[1].startswith("\x1b_Gq=2,m=1;")
    assert chunks[-1].startswith("\x1b_Gq=2,m=0;")
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


def test_show_image_flattens_transparent_png():
    data = png_bytes(mode="RGBA", color=(0, 0, 0, 0))
    out = io.StringIO()
    tan_display.show_image(data, mode="kitty", out=out)
    payload = "".join(c.split(";", 1)[1] for c in out.getvalue().split("\x1b\\") if c.strip())
    image = Image.open(io.BytesIO(base64.b64decode(payload)))
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(("mode", "flag"), [("kitty", ",c=19,"), ("iterm", ";width=19;")])
def test_show_image_caps_graphics_at_terminal_size(mode, flag, monkeypatch):
    monkeypatch.setattr(tan_display.shutil, "get_terminal_size", lambda fallback: os.terminal_size((20, 30)))
    out = io.StringIO()
    tan_display.show_image(png_bytes(size=(400, 100)), mode=mode, out=out)  # 40 cells at 10 px, 19 fit
    assert flag in out.getvalue()


def test_show_image_leaves_small_graphics_at_native_size(monkeypatch):
    monkeypatch.setattr(tan_display.shutil, "get_terminal_size", lambda fallback: os.terminal_size((80, 24)))
    out = io.StringIO()
    tan_display.show_image(png_bytes(), mode="kitty", out=out)
    assert out.getvalue().startswith("\x1b_Ga=T,f=100,q=2,m=")


def stripes(module, count=30):
    image = Image.new("RGB", (module * count, module * 4), "white")
    for x in range(image.width):
        if (x // module) % 2:
            for y in range(image.height):
                image.putpixel((x, y), (0, 0, 0))
    return image


@pytest.mark.parametrize("module", [1, 3, 5])
def test_module_size(module):
    assert tan_display.module_size(stripes(module)) == module


def test_module_size_ignores_stray_pixels():
    image = stripes(4, count=60)
    for x, y in [(41, 2), (101, 7), (161, 12)]:  # single light pixels inside dark modules
        image.putpixel((x, y), (255, 255, 255))
    assert tan_display.module_size(image) == 4


def test_blocks_use_wide_terminals(monkeypatch):
    """No fixed column cap, so enlarging the terminal as the too-small warning says actually helps."""
    monkeypatch.setattr(tan_display.shutil, "get_terminal_size", lambda fallback: os.terminal_size((201, 202)))
    monkeypatch.setenv("COLORTERM", "truecolor")
    out = io.StringIO()
    tan_display.show_image(png_bytes(size=(400, 400), color=(255, 255, 255)), mode="blocks", out=out)
    first_line = out.getvalue().splitlines()[0]
    assert first_line.count("▀") == 200


@pytest.mark.parametrize(
    ("env", "depth"),
    [
        ({"COLORTERM": "truecolor", "TERM": "xterm-256color"}, "truecolor"),
        ({"TERM": "xterm-kitty"}, "truecolor"),
        ({"TERM": "xterm-256color"}, "256"),
        ({"TERM": "linux"}, "16"),
        ({}, "16"),
    ],
)
def test_color_depth(env, depth):
    assert tan_display.color_depth(env) == depth


@pytest.mark.parametrize(
    ("depth", "black", "white"),
    [
        ("truecolor", "\x1b[38;2;0;0;0m", "\x1b[48;2;255;255;255m"),
        ("256", "\x1b[38;5;16m", "\x1b[48;5;231m"),
        ("16", "\x1b[30m", "\x1b[107m"),
    ],
)
def test_sgr_color_keeps_black_and_white_exact(depth, black, white):
    assert tan_display.sgr_color((0, 0, 0), background=False, depth=depth) == black
    assert tan_display.sgr_color((255, 255, 255), background=True, depth=depth) == white


def test_ansi256_maps_colors_to_the_cube_and_grays_to_the_ramp():
    assert tan_display._ansi256((255, 0, 0)) == 196
    assert tan_display._ansi256((128, 128, 128)) == 244


def test_block_art_without_truecolor_has_no_24bit_codes():
    art = tan_display.block_art(Image.new("RGB", (4, 4), (200, 30, 30)), columns=4, depth="256")
    assert "38;2;" not in art and "38;5;" in art


@pytest.mark.parametrize(("scale", "warns"), [(1.0, False), (0.5, False), (0.3, True)])
def test_warn_if_too_small(scale, warns, monkeypatch):
    messages = []
    monkeypatch.setattr(tan_display.logger, "warning", lambda msg, *a: messages.append(msg % a))
    tan_display.warn_if_too_small(stripes(3), scale)
    assert bool(messages) == warns
    if warns:
        assert "31 columns" in messages[0]


def test_show_image_rejects_broken_image():
    with pytest.raises(tan_display.ImageDecodeError):
        tan_display.show_image(b"junk", mode="blocks", out=io.StringIO())


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


@pytest.mark.parametrize(
    "reply",
    ["\x1b_Gi=31;OK\x1b\\\x1b[?62;c", "\x1b_Gi=1;ENOENT:no such image\x1b\\", "\x1b[?64;1;2c", "\x1b]1337;x\x07"],
)
def test_read_line_strips_terminal_replies(reply, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "12" + reply + "3456")
    assert tan_display.read_line("TAN: ") == "123456"


@pytest.mark.parametrize("garbled", ["\x1b_Gi=31;OK123456", "\x1b[?62;123456", "\x1b123456"])
def test_read_line_asks_again_after_cut_reply(garbled, monkeypatch):
    lines = iter([garbled, "123456"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    assert tan_display.read_line("TAN: ") == "123456"


def test_show_window_needs_a_display(monkeypatch):
    monkeypatch.setattr(tan_display.sys, "platform", "linux")
    monkeypatch.setattr(tan_display.ImageShow, "show", lambda image: pytest.fail("viewer started"))
    with pytest.raises(OSError, match="no graphical display"):
        tan_display.show_window(Image.new("RGB", (2, 2)), env={})


def test_show_window_fails_without_viewer(monkeypatch):
    monkeypatch.setattr(tan_display.sys, "platform", "linux")
    monkeypatch.setattr(tan_display.ImageShow, "show", lambda image: False)
    with pytest.raises(OSError, match="no image viewer"):
        tan_display.show_window(Image.new("RGB", (2, 2)), env={"DISPLAY": ":0"})


def test_configured_mode_keeps_auto():
    assert tan_display.configured_mode("auto", {}) == "auto"
    assert tan_display.configured_mode("auto", {"CLEANAB_TAN_DISPLAY": "Kitty"}) == "kitty"


def test_resolve_once_probes_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(tan_display, "_auto_mode", lambda env: calls.append(env) or "blocks")
    assert tan_display.resolve_once("auto") == "blocks"
    assert tan_display.resolve_once("auto") == "blocks"
    assert tan_display.resolve_once("iterm") == "iterm"
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["sixel", "auto"])
def test_unknown_or_unresolved_mode_rejected(mode):
    with pytest.raises(ValueError, match="Unknown TAN display mode"):
        tan_display.show_image(png_bytes(), mode=mode, out=io.StringIO())


def matrix_response():
    return SimpleNamespace(
        challenge="scan", challenge_hhduc=None, challenge_matrix=("image/png", png_bytes()), challenge_html=""
    )


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(fints_module.sys.stdout, "isatty", lambda: True, raising=False)


def test_handle_tan_response_shows_matrix_image(monkeypatch, tty):
    shown = []
    monkeypatch.setattr(fints_module, "show_image", lambda data, mode, **kw: shown.append((data, mode)))
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    client = SimpleNamespace(send_tan=lambda r, tan: ["tx"], tan_display="kitty")
    assert fints_module.handle_tan_response(client, matrix_response()) == ["tx"]
    assert shown == [(png_bytes(), "kitty")]


def test_handle_tan_response_survives_display_failure(monkeypatch, tty):
    attempts = []

    def broken(data, mode, **kw):
        attempts.append(mode)
        raise OSError("cannot write mode CMYK as PNG")

    monkeypatch.setattr(fints_module, "show_image", broken)
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    client = SimpleNamespace(send_tan=lambda r, tan: ["tx"], tan_display="kitty")
    assert fints_module.handle_tan_response(client, matrix_response()) == ["tx"]
    assert attempts == ["kitty", "blocks"]


def test_show_tan_image_tries_blocks_once(monkeypatch, tty):
    attempts = []

    def broken(data, mode, **kw):
        attempts.append(mode)
        raise OSError("broken pipe")

    monkeypatch.setattr(fints_module, "show_image", broken)
    fints_module.show_tan_image(png_bytes(), "blocks")
    assert attempts == ["blocks"]


def test_show_tan_image_does_not_retry_decode_errors(monkeypatch, tty):
    errors = []
    monkeypatch.setattr(fints_module.logger, "exception", lambda *a: errors.append("traceback"))
    monkeypatch.setattr(fints_module.logger, "error", lambda *a: errors.append("error"))
    monkeypatch.setattr(fints_module, "show_image", lambda *a, **kw: pytest.fail("shown"))
    fints_module.show_tan_image(b"junk", "kitty")
    assert errors == ["error"]


def test_show_tan_image_expected_failure_has_no_traceback(monkeypatch, tty):
    logged = []

    def broken(data, mode, **kw):
        raise OSError("no graphical display")

    monkeypatch.setattr(fints_module, "show_image", broken)
    monkeypatch.setattr(fints_module.logger, "exception", lambda *a: logged.append("traceback"))
    monkeypatch.setattr(fints_module.logger, "warning", lambda *a: logged.append("warning"))
    monkeypatch.setattr(tan_display.sys.stdout, "write", lambda text: len(text), raising=False)
    fints_module.show_tan_image(png_bytes(), "window")
    assert logged == ["warning", "warning"]


def test_show_tan_image_decodes_once(monkeypatch, tty):
    decodes = []
    real_decode = fints_module.decode
    monkeypatch.setattr(fints_module, "decode", lambda data: decodes.append(1) or real_decode(data))
    monkeypatch.setattr(tan_display, "decode", lambda data: pytest.fail("decoded again"))
    attempts = []

    def flaky(data, mode, **kw):
        attempts.append(mode)
        if mode == "kitty":
            raise OSError("broken")
        return tan_display.show_image(data, mode, out=io.StringIO(), **kw)

    monkeypatch.setattr(fints_module, "show_image", flaky)
    fints_module.show_tan_image(png_bytes(), "kitty")
    assert attempts == ["kitty", "blocks"]
    assert decodes == [1]


def test_show_tan_image_resolves_auto(monkeypatch, tty):
    shown = []
    monkeypatch.setattr(fints_module, "resolve_once", lambda mode: "iterm")
    monkeypatch.setattr(fints_module, "show_image", lambda data, mode, **kw: shown.append(mode))
    fints_module.show_tan_image(png_bytes(), "auto")
    assert shown == ["iterm"]


def test_show_tan_image_window_without_tty_does_not_fall_back(monkeypatch):
    attempts = []

    def broken(data, mode, **kw):
        attempts.append(mode)
        raise OSError("no graphical display")

    monkeypatch.setattr(fints_module.sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(fints_module, "show_image", broken)
    fints_module.show_tan_image(png_bytes(), "window")
    assert attempts == ["window"]


@pytest.mark.parametrize(("mode", "shown"), [("blocks", False), ("window", True)])
def test_show_tan_image_skips_terminal_modes_without_tty(mode, shown, monkeypatch):
    calls = []
    monkeypatch.setattr(fints_module.sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(fints_module, "show_image", lambda data, m, **kw: calls.append(m))
    fints_module.show_tan_image(png_bytes(), mode)
    assert bool(calls) == shown
