import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from cleanab import fints as fints_module
from cleanab import tan_display


def png_bytes(size=(8, 8), color=(0, 0, 0)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"TERM": "xterm-ghostty"}, "kitty"),
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
        ({"TERM": "xterm-kitty", "TMUX": "/tmp/tmux"}, "blocks"),
        ({"TERM": "screen-256color"}, "blocks"),
        ({"SSH_CONNECTION": "a b c d", "DISPLAY": ":0"}, "blocks"),
        ({"DISPLAY": ":0"}, "window"),
        ({}, "blocks"),
    ],
)
def test_detect_mode(env, expected, monkeypatch):
    monkeypatch.setattr(tan_display.sys, "platform", "linux")
    assert tan_display.detect_mode(env) == expected


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
    assert "\x1b[38;2;0;0;0m\x1b[48;2;255;255;255m▀" in art[0]
    scaled = tan_display.block_art(Image.new("RGB", (100, 100)), columns=20).splitlines()
    assert len(scaled) == 10


def test_show_image_writes_selected_mode():
    out = io.StringIO()
    assert tan_display.show_image(png_bytes(), mode="kitty", out=out) == "kitty"
    assert out.getvalue().startswith("\x1b_Ga=T")
    out = io.StringIO()
    assert tan_display.show_image(png_bytes(), mode="blocks", out=out) == "blocks"
    assert "▀" in out.getvalue()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unknown TAN display mode"):
        tan_display.show_image(png_bytes(), mode="sixel", out=io.StringIO())


def test_environment_overrides_configured_mode(monkeypatch):
    monkeypatch.setenv("CLEANAB_TAN_DISPLAY", "iterm")
    out = io.StringIO()
    assert tan_display.show_image(png_bytes(), out=out) == "iterm"


def test_handle_tan_response_shows_matrix_image(monkeypatch):
    shown = []
    monkeypatch.setattr(fints_module, "show_image", lambda data: shown.append(data) or "blocks")
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    response = SimpleNamespace(
        challenge="scan", challenge_hhduc=None, challenge_matrix=("image/png", b"PNGDATA"), challenge_html=""
    )
    client = SimpleNamespace(send_tan=lambda r, tan: ["tx"])
    assert fints_module.handle_tan_response(client, response) == ["tx"]
    assert shown == [b"PNGDATA"]
