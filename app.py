#!/usr/bin/env python3
"""
A plain one-page web UI for video_watcher: paste a link, tap Watch, read the
result. No notebook, no cells.

Run it locally:
    python app.py            (or double-click run.bat on Windows)
It opens a clean page in your browser and also prints a public link you can
open on your phone.

It also runs as-is on Hugging Face Spaces (set ANTHROPIC_API_KEY as a Space
secret and make the Space private so only you can use your key).
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "video_watcher.py"
KEY_FILE = HERE / ".vw_key"          # local, gitignored: so you paste the key once


def _load_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def watch(url, uploaded, effort, social, audio_only, api_key):
    key = (api_key or "").strip() or _load_key()
    if not key.startswith("sk-ant-"):
        yield ("Paste your Anthropic API key (it starts with `sk-ant-`) in the "
               "key box, then tap Watch. You only do this once on this device.")
        return
    # Remember it on this machine so it does not have to be retyped.
    try:
        KEY_FILE.write_text(key, encoding="utf-8")
    except OSError:
        pass

    source = uploaded or (url or "").strip()
    if not source:
        yield "Paste a video link, or choose a file, first."
        return

    out = HERE / "runs" / uuid.uuid4().hex[:8]
    cmd = [sys.executable, str(SCRIPT), str(source), "--out", str(out)]
    cmd += {"Fast": ["--fast"], "Thorough": ["--thorough"]}.get(effort, [])
    if social:
        cmd.append("--social")
    if audio_only:
        cmd.append("--audio-only")
    env = dict(os.environ, ANTHROPIC_API_KEY=key, PYTHONUNBUFFERED="1")

    # Stream the tool's own log lines to the page so it never looks frozen.
    yield ("Starting… the first run downloads a small speech model, so the very "
           "first one can take a minute. Progress shows below.")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    logs = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logs.append(line)
            yield "Working…\n\n```\n" + "\n".join(logs[-18:]) + "\n```"
    proc.wait()

    result = out / "understanding.md"
    if result.exists():
        yield result.read_text(encoding="utf-8")
    else:
        tail = "\n".join(logs[-40:]) or "no output"
        yield ("It did not finish. Here is what it reported:\n\n```\n"
               + tail + "\n```")


with gr.Blocks(title="Video Watcher") as demo:
    gr.Markdown(
        "# 🎬 Video Watcher\n"
        "Paste a TikTok, YouTube, or Instagram link (or a podcast / mp3 link) and "
        "tap **Watch**. You get a plain-language recap, and (with the caption box "
        "on) a ready-to-post caption and hashtags."
    )
    key_box = gr.Textbox(
        label="Anthropic API key — paste once",
        type="password",
        placeholder="sk-ant-...  (saved only on this device, never shared)",
    )
    url_box = gr.Textbox(
        label="Video or audio link",
        placeholder="https://...  (TikTok, YouTube, Instagram, podcast, mp3)",
    )
    file_box = gr.File(label="…or upload a video or audio file instead",
                       type="filepath")
    with gr.Row():
        effort = gr.Radio(
            ["Fast", "Normal", "Thorough"], value="Fast",
            label="Effort (Fast is cheap and quick; Thorough is the most careful)",
        )
        social = gr.Checkbox(value=True, label="Add caption + hashtags")
        audio_only = gr.Checkbox(value=False, label="Audio only (skip visuals)")
    go = gr.Button("Watch", variant="primary")
    out_md = gr.Markdown(label="Result")
    go.click(watch, [url_box, file_box, effort, social, audio_only, key_box],
             out_md)


def _primary_ip():
    """The address this PC uses to reach the network — what a phone on the same
    Wi-Fi or hotspot should open."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):           # running on Hugging Face Spaces
        demo.launch()
    else:
        ip = _primary_ip()
        print("\n" + "=" * 62)
        print("  Open Video Watcher:")
        print("    On this PC:                          http://127.0.0.1:7860")
        if ip:
            print(f"    On your phone (same Wi-Fi/hotspot):  http://{ip}:7860")
        print("  First time: if Windows shows a firewall box, click Allow access.")
        print("  If the phone link will not load, the network blocks")
        print("  device-to-device traffic; use the Hugging Face Space instead.")
        print("=" * 62 + "\n")
        # Bind to the whole network so a phone on the same Wi-Fi can reach it.
        demo.launch(server_name="0.0.0.0", server_port=7860)
