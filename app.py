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


def watch(url, uploaded, effort, social, api_key):
    key = (api_key or "").strip() or _load_key()
    if not key.startswith("sk-ant-"):
        return ("Paste your Anthropic API key (it starts with `sk-ant-`) in the "
                "key box, then tap Watch. You only do this once on this device.")
    # Remember it on this machine so it does not have to be retyped.
    try:
        KEY_FILE.write_text(key, encoding="utf-8")
    except OSError:
        pass

    source = uploaded or (url or "").strip()
    if not source:
        return "Paste a video link, or choose a file, first."

    out = HERE / "runs" / uuid.uuid4().hex[:8]
    cmd = [sys.executable, str(SCRIPT), str(source), "--out", str(out)]
    cmd += {"Fast": ["--fast"], "Thorough": ["--thorough"]}.get(effort, [])
    if social:
        cmd.append("--social")
    env = dict(os.environ, ANTHROPIC_API_KEY=key)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    result = out / "understanding.md"
    if result.exists():
        return result.read_text(encoding="utf-8")
    tail = (proc.stderr or proc.stdout or "no output").strip()[-2500:]
    return "It did not finish. Here is what it reported:\n\n```\n" + tail + "\n```"


with gr.Blocks(title="Video Watcher") as demo:
    gr.Markdown(
        "# 🎬 Video Watcher\n"
        "Paste a TikTok, YouTube, or Instagram link and tap **Watch**. "
        "You get a plain-language recap, and (with the caption box on) a "
        "ready-to-post caption and hashtags."
    )
    key_box = gr.Textbox(
        label="Anthropic API key — paste once",
        type="password",
        placeholder="sk-ant-...  (saved only on this device, never shared)",
    )
    url_box = gr.Textbox(label="Video link", placeholder="https://...")
    file_box = gr.File(label="…or upload a video file instead", type="filepath")
    with gr.Row():
        effort = gr.Radio(
            ["Fast", "Normal", "Thorough"], value="Fast",
            label="Effort (Fast is cheap and quick; Thorough is the most careful)",
        )
        social = gr.Checkbox(value=True, label="Add caption + hashtags")
    go = gr.Button("Watch", variant="primary")
    out_md = gr.Markdown(label="Result")
    go.click(watch, [url_box, file_box, effort, social, key_box], out_md)


def _lan_ips():
    """All this machine's IPv4 addresses, so we can show the right one to type."""
    import socket
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    ips.discard("127.0.0.1")
    return sorted(ips)


def _is_home_lan(ip: str) -> bool:
    return (ip.startswith("192.168.") or ip.startswith("10.")
            or any(ip.startswith(f"172.{n}.") for n in range(16, 32)))


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):           # running on Hugging Face Spaces
        demo.launch()
    else:
        ips = _lan_ips()
        home = [ip for ip in ips if _is_home_lan(ip)]
        tailscale = [ip for ip in ips if ip.startswith("100.")]
        print("\n" + "=" * 62)
        print("  Open Video Watcher:")
        print("    On this PC:                  http://127.0.0.1:7860")
        for ip in home:
            print(f"    On your phone (same Wi-Fi):  http://{ip}:7860")
        for ip in tailscale:
            print(f"    Via Tailscale (phone needs Tailscale on too):  "
                  f"http://{ip}:7860")
        if not home and not tailscale:
            for ip in ips:
                print(f"    Try on your phone:           http://{ip}:7860")
        print("  If Windows shows a firewall box, click Allow access.")
        print("=" * 62 + "\n")
        # Bind to the whole network so a phone on the same Wi-Fi can reach it.
        demo.launch(server_name="0.0.0.0", server_port=7860)
