#!/usr/bin/env python3
"""
video_watcher.py

Watch and "understand" a video the way an attentive human viewer would.

The pipeline has four stages:

  1. LISTEN   Extract the audio track and transcribe speech with faster-whisper,
              keeping per-segment timestamps.
  2. SEE      Extract still frames with ffmpeg. Frames are pulled both at scene
              changes (so cuts are captured) and on a steady cadence (so slow,
              static shots are still covered), then de-duplicated and capped.
  3. ANALYZE  Slice the video into short time windows. For each window, send the
              frames in that window plus the spoken transcript for that window to
              Claude's vision model, and ask it to describe what is happening.
              This is the "map" step. Windows are analyzed concurrently.
  4. SYNTHESIZE  Feed every window observation plus the full transcript to Claude
              and ask for one faithful, human-style account of the whole video.
              This is the "reduce" step.

Outputs (written into an output folder named after the video):
  transcript.txt      full transcript with timestamps
  transcript.json     raw transcript segments
  observations.json   per-window descriptions
  understanding.md    the final synthesis (the main deliverable)
  meta.json           run parameters and counts
  frames/             the extracted JPEG frames

Requirements:
  - ffmpeg and ffprobe on PATH
      Linux:   sudo apt install ffmpeg
      macOS:   brew install ffmpeg
      Windows: winget install Gyan.FFmpeg  (or choco/scoop install ffmpeg)
  - pip install -r requirements.txt        (anthropic, faster-whisper)
  - set your Anthropic API key:
      macOS/Linux:        export ANTHROPIC_API_KEY=sk-ant-...
      Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
      Windows cmd:        set ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python video_watcher.py myvideo.mp4
  python video_watcher.py talk.mp4 --window 20 --map-model claude-sonnet-4-6
  python video_watcher.py long.mp4 --start 60 --end 180 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def log(stage: str, msg: str) -> None:
    """Print a timestamped progress line and flush immediately."""
    print(f"[{stage}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def format_ts(seconds: float) -> str:
    """Turn 75.4 into 01:15 and 3725 into 1:02:05."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run(cmd: list[str], capture_stdout: bool = False) -> subprocess.CompletedProcess:
    """
    Run a command. stderr is always captured (ffmpeg writes showinfo there).
    Raises CalledProcessError on a non-zero exit.

    Output is decoded as UTF-8 with errors='replace'. ffmpeg emits UTF-8 on
    stderr; without this, decoding falls back to the OS locale codec (cp1252
    on Windows) and a single non-Latin byte in the video's metadata would raise
    UnicodeDecodeError inside subprocess.run, which main() cannot catch.
    """
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def check_binaries() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            die(
                f"'{tool}' was not found on your PATH. Install ffmpeg first.\n"
                f"  Linux:   sudo apt install ffmpeg\n"
                f"  macOS:   brew install ffmpeg\n"
                f"  Windows: winget install Gyan.FFmpeg  "
                f"(or choco/scoop install ffmpeg)"
            )


def probe_duration(path: Path) -> float:
    """Return the media duration in seconds using ffprobe."""
    out = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_stdout=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        die(f"Could not read the duration of {path}. Is it a valid media file?")
        return 0.0  # unreachable, keeps type checkers happy


def is_url(s: str) -> bool:
    """True for an http(s) URL, False for any local path (incl. Windows C:\\...)."""
    from urllib.parse import urlparse
    u = urlparse(str(s))
    return u.scheme in ("http", "https") and bool(u.netloc)


def download_url(url: str, source_dir: Path) -> Path:
    """Download a single video to source_dir with yt-dlp and return its path.
    yt-dlp is an optional dependency, only needed when the input is a URL.
    Reuses the ffmpeg already on PATH to merge the best video + audio."""
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("yt-dlp")
    base = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd = base + [
        "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
        "-o", str(source_dir / "%(id)s.%(ext)s"),
        "--print", "after_move:filepath", "--no-warnings", "--restrict-filenames",
        url,
    ]
    log("setup", f"downloading {url} with yt-dlp")
    try:
        proc = run(cmd, capture_stdout=True)
    except FileNotFoundError:
        die("yt-dlp is not installed. Run: pip install -U yt-dlp")
        return source_dir  # unreachable
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        if "No module named" in err and "yt_dlp" in err:
            die("yt-dlp is not installed. Run: pip install -U yt-dlp")
        die(f"yt-dlp failed to download {url}:\n{err[-1500:]}")
    # yt-dlp prints the final path; trust it when the file exists.
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line and Path(line).exists():
            return Path(line)
    # Fallback (the printed extension can differ after a remux): take the
    # largest finished media file in the dedicated, freshly-emptied dir.
    media = {".mp4", ".mkv", ".webm", ".m4a", ".mov"}
    files = [p for p in source_dir.iterdir()
             if p.is_file() and p.suffix.lower() in media
             and not p.name.endswith((".part", ".ytdl"))]
    if not files:
        die(f"yt-dlp produced no media file for {url}.")
        return source_dir  # unreachable
    return max(files, key=lambda p: p.stat().st_size)


# Rough USD per 1M tokens (input, output) for the dry-run cost estimate only.
_PRICING = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class Config:
    video: Path
    out_dir: Path

    # trimming (optional). times are in seconds, relative to the original video.
    start: Optional[float] = None
    end: Optional[float] = None

    # audio / transcription
    no_audio: bool = False
    whisper_model: str = "base"          # tiny, base, small, medium, large-v3
    device: str = "cpu"                  # cpu or cuda
    compute_type: str = "int8"           # int8 (cpu), float16 (gpu), etc.
    language: Optional[str] = None       # None = auto detect

    # frame extraction
    use_scene: bool = True
    scene_threshold: float = 0.20        # 0..1, lower catches subtler motion
    fps_interval: float = 3.0            # fill cadence used only across long static gaps
    max_gap: float = 6.0                 # never leave a stretch longer than this unsampled
    min_gap: float = 1.5                 # do not keep frames closer together than this
    max_frames: int = 120                # overall cap across the whole video
    frame_width: int = 768               # downscale long edge to this many px
    jpeg_q: int = 4                      # ffmpeg -q:v, 2 best .. 31 worst

    # windowing / models
    window: float = 30.0                 # seconds of video per analysis window
    max_frames_per_window: int = 10
    map_model: str = "claude-haiku-4-5-20251001"
    reduce_model: str = "claude-sonnet-4-6"
    map_max_tokens: int = 700
    reduce_max_tokens: int = 4000
    concurrency: int = 4                 # parallel vision (map) calls

    # behavior
    dry_run: bool = False                # extract + transcribe, skip all API calls
    resume: bool = False                 # reuse completed observations from a prior run
    social: bool = False                 # also emit creator hook/caption/hashtags
    source_url: Optional[str] = None     # original URL, if the input was downloaded


# ----------------------------------------------------------------------------
# Stage 1: audio + transcription
# ----------------------------------------------------------------------------

def extract_audio(video: Path, out_wav: Path) -> None:
    """Pull a 16 kHz mono PCM wav, which is what Whisper wants."""
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_wav),
    ])


class Transcriber:
    """Thin wrapper around faster-whisper that returns timestamped segments."""

    def __init__(self, model_size: str, device: str, compute_type: str,
                 language: Optional[str]):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            die("faster-whisper is not installed. Run: pip install faster-whisper")
        log("listen", f"loading whisper model '{self.model_size}' "
                       f"({self.device}/{self.compute_type})")
        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as e:
            die(f"failed to load whisper model '{self.model_size}' "
                f"on {self.device}/{self.compute_type}: {e}\n"
                f"check the model size (tiny/base/small/medium/large-v3) and, "
                f"for --device cuda, that CUDA and cuDNN are installed.")

    def transcribe(self, audio_path: Path, time_offset: float = 0.0) -> list[dict]:
        self._load()
        segments: list[dict] = []
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=True,                 # skip silence
                beam_size=5,
            )
            detected = getattr(info, "language", None)
            if detected:
                log("listen", f"detected language: {detected} "
                              f"(p={getattr(info, 'language_probability', 0):.2f})")
            # faster-whisper decodes lazily as we iterate, so errors surface here.
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append({
                    "start": round(seg.start + time_offset, 3),
                    "end": round(seg.end + time_offset, 3),
                    "text": text,
                })
        except Exception as e:
            die(f"transcription failed: {e}")
        log("listen", f"transcribed {len(segments)} speech segments")
        return segments


# ----------------------------------------------------------------------------
# Stage 2: frame extraction
# ----------------------------------------------------------------------------

class FrameExtractor:
    """
    Extracts timestamped frames using ffmpeg. Each ffmpeg pass uses the
    'showinfo' filter, whose stderr output lists the presentation timestamp
    (pts_time) of every frame that passes through, in order. We zip those
    timestamps with the JPEG files ffmpeg writes (also in order) to recover an
    accurate (timestamp, image_path) pair for each frame.
    """

    PTS_RE = re.compile(r"pts_time:(\S+)")

    def __init__(self, frames_dir: Path, frame_width: int, jpeg_q: int):
        self.frames_dir = frames_dir
        self.frame_width = frame_width
        self.jpeg_q = jpeg_q
        # Start clean: a deterministic out folder is reused across runs, and
        # stale higher-numbered JPEGs from a previous run would otherwise be
        # globbed and mis-timestamped. Only the frames/ subdir is wiped; the
        # transcript, audio, and meta in the parent work dir are left alone.
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _parse_pts(cls, stderr: str) -> list:
        """One entry per showinfo frame, in order. Non-numeric pts (N/A) become
        None so the list stays 1:1 with the frames it describes."""
        out = []
        for raw in cls.PTS_RE.findall(stderr or ""):
            try:
                out.append(float(raw))
            except ValueError:
                out.append(None)
        return out

    def _scale_clause(self) -> str:
        # keep aspect ratio, cap the long edge, force even dimensions.
        # force_original_aspect_ratio=decrease caps BOTH edges to frame_width
        # (so portrait video is sized by its height, not just its width).
        return (f"scale=w={self.frame_width}:h={self.frame_width}:"
                f"force_original_aspect_ratio=decrease:force_divisible_by=2")

    def _pair(self, files: list, pts: list, time_offset: float) -> list:
        """Pair sorted JPEG files with parsed pts_time values, 1:1, carrying
        the last good timestamp forward across any N/A."""
        pairs: list[tuple[float, Path]] = []
        last = 0.0
        n = min(len(files), len(pts))
        for i in range(n):
            t = pts[i]
            if t is None:
                t = last
            last = t
            pairs.append((round(t + time_offset, 3), files[i]))
        return pairs

    def _pass(self, video: Path, select_filter: str, label: str,
              time_offset: float) -> list[tuple[float, Path]]:
        out_pattern = self.frames_dir / f"{label}_%05d.jpg"
        vf = f"{select_filter},{self._scale_clause()},showinfo"
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", vf,
            "-fps_mode", "vfr",              # do not pad to a constant frame rate
            "-q:v", str(self.jpeg_q),
            str(out_pattern),
        ]
        proc = run(cmd)
        pts = self._parse_pts(proc.stderr or "")
        files = sorted(self.frames_dir.glob(f"{label}_*.jpg"))
        if len(pts) != len(files):
            log("see", f"warning: {label} pts count ({len(pts)}) != frame "
                       f"count ({len(files)}); timestamps may be approximate")
        return self._pair(files, pts, time_offset)

    def _combined_pass(self, video: Path, scene_filter: str, rate: float,
                       time_offset: float) -> tuple[list, list]:
        """Decode the video once and split it into a scene branch and a grid
        branch, instead of decoding twice. Only the scene branch carries a
        showinfo (its frame times are data-dependent); grid frame times are
        deterministic (i / rate) and computed directly."""
        change_pattern = self.frames_dir / "change_%05d.jpg"
        grid_pattern = self.frames_dir / "grid_%05d.jpg"
        scale = self._scale_clause()
        filter_complex = (f"[0:v]{scale},split=2[a][b];"
                          f"[a]{scene_filter},showinfo[scn];"
                          f"[b]fps={rate:.6f}[grd]")
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-filter_complex", filter_complex,
            "-map", "[scn]", "-fps_mode", "vfr", "-q:v", str(self.jpeg_q),
            str(change_pattern),
            "-map", "[grd]", "-fps_mode", "vfr", "-q:v", str(self.jpeg_q),
            str(grid_pattern),
        ]
        proc = run(cmd)
        pts = self._parse_pts(proc.stderr or "")   # all from the scene branch
        change_files = sorted(self.frames_dir.glob("change_*.jpg"))
        if len(pts) != len(change_files):
            log("see", f"warning: change pts count ({len(pts)}) != frame "
                       f"count ({len(change_files)}); timestamps may be approximate")
        change = self._pair(change_files, pts, time_offset)
        grid_files = sorted(self.frames_dir.glob("grid_*.jpg"))
        grid = [(round(i / rate + time_offset, 3), f)
                for i, f in enumerate(grid_files)]
        return change, grid

    def extract(self, video: Path, cfg: Config, duration: float,
                time_offset: float) -> list[tuple[float, Path]]:
        # Two candidate sets feed the merge below, which decides what to keep
        # the way a viewer's attention works: linger where the picture is
        # changing, glance periodically where it is static.

        # Candidate set A: "change" frames (cuts and visible motion, the moments
        # the eye catches). Candidate set B: an evenly spaced grid used only to
        # bridge long static stretches. When both are wanted we decode once.
        change: list[tuple[float, Path]] = []
        grid: list[tuple[float, Path]] = []
        scene_filter = f"select='gt(scene,{cfg.scene_threshold})+eq(n,0)'"
        want_scene = cfg.use_scene
        want_grid = bool(cfg.fps_interval and cfg.fps_interval > 0)

        if want_scene and want_grid:
            rate = 1.0 / cfg.fps_interval
            change, grid = self._combined_pass(video, scene_filter, rate, time_offset)
            log("see", f"found {len(change)} change frames "
                       f"(threshold {cfg.scene_threshold:g})")
        elif want_scene:
            change = self._pass(video, scene_filter, "change", time_offset)
            log("see", f"found {len(change)} change frames "
                       f"(threshold {cfg.scene_threshold:g})")
        elif want_grid:
            rate = 1.0 / cfg.fps_interval
            grid = self._pass(video, f"fps={rate:.6f}", "grid", time_offset)

        if not change and not grid:
            change = self._pass(video, "select='eq(n,0)'", "change", time_offset)

        start_t = time_offset
        end_t = time_offset + duration

        # Step 1: thin out clustered change frames so one busy moment does not
        # eat the whole budget (min_gap floor).
        change.sort(key=lambda x: x[0])
        kept: list[tuple[float, Path]] = []
        last_t = -1e9
        for t, p in change:
            if t - last_t >= cfg.min_gap:
                kept.append((t, p))
                last_t = t

        # Step 2: bridge gaps. Wherever two kept frames (or the clip edges) are
        # more than max_gap apart, drop in grid frames spaced about fps_interval
        # apart, so static stretches are still sampled periodically.
        def fills_between(lo: float, hi: float) -> list[tuple[float, Path]]:
            if hi - lo <= cfg.max_gap:
                return []
            out: list[tuple[float, Path]] = []
            last = lo
            for t, p in grid:
                if lo < t < hi and t - last >= cfg.fps_interval:
                    out.append((t, p))
                    last = t
            return out

        bridged = list(kept)
        anchors = [start_t] + [t for t, _ in kept] + [end_t]
        for lo, hi in zip(anchors, anchors[1:]):
            bridged.extend(fills_between(lo, hi))

        # Step 3: final sort and min_gap pass to clean up any new neighbours.
        bridged.sort(key=lambda x: x[0])
        merged: list[tuple[float, Path]] = []
        last_t = -1e9
        for t, p in bridged:
            if t - last_t >= cfg.min_gap:
                merged.append((t, p))
                last_t = t

        # Step 4: hard cap with an even subsample across the timeline.
        if len(merged) > cfg.max_frames:
            step = len(merged) / cfg.max_frames
            idx = sorted({int(i * step) for i in range(cfg.max_frames)})
            merged = [merged[i] for i in idx if i < len(merged)]

        # Step 5: delete extracted frames we are not using, so the frames folder
        # matches exactly what was sent for analysis.
        keep_paths = {p for _t, p in merged}
        for f in self.frames_dir.glob("*.jpg"):
            if f not in keep_paths:
                f.unlink(missing_ok=True)

        log("see", f"keeping {len(merged)} frames "
                   f"(dense where the video changes, periodic where it is static)")
        return merged


# ----------------------------------------------------------------------------
# Stage 3 + 4: Claude vision analysis and synthesis
# ----------------------------------------------------------------------------

def encode_image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


MAP_SYSTEM = (
    "You are watching a video the way an attentive human viewer would. "
    "You are given a short window of the video as a few still frames in time "
    "order, each labeled with its timestamp, along with the transcript of what "
    "is spoken during that window. Study each frame closely, as if you paused "
    "on it: read any on-screen text and notice small but meaningful details. "
    "Then describe what is actually happening across the window: the "
    "setting, who or what is on screen, actions and movement, any on-screen "
    "text or graphics, and how the visuals line up with the words. Note the "
    "apparent tone or mood. Describe only what the frames and transcript "
    "support. If something is unclear, say so rather than inventing detail. "
    "Do not use em dashes anywhere in your reply; use commas or rewrite."
)

REDUCE_SYSTEM = (
    "You have watched an entire video by reviewing time-ordered observations of "
    "each segment together with the full transcript. Give one clear, faithful "
    "account of the video as a thoughtful person would after watching it once, "
    "closely. Stay grounded in the observations and transcript and do not invent "
    "details. Do not use em dashes anywhere in your reply; use commas or rewrite."
)


class VisionAnalyzer:
    """Map step: describe each time window from its frames and spoken words."""

    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        try:
            import anthropic
        except ImportError:
            die("The anthropic package is not installed. Run: pip install anthropic")
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5)

    def analyze_window(self, idx: int, w_start: float, w_end: float,
                       frames: list[tuple[float, Path]],
                       transcript_text: str) -> str:
        content: list[dict] = []
        for t, path in frames:
            content.append({"type": "text", "text": f"Frame at {format_ts(t)}:"})
            content.append(encode_image_block(path))

        spoken = transcript_text.strip() or "(no speech detected in this window)"
        content.append({
            "type": "text",
            "text": (
                f"Transcript for this window "
                f"({format_ts(w_start)} to {format_ts(w_end)}):\n{spoken}\n\n"
                f"In 2 to 5 sentences, describe what happens in this window."
            ),
        })

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=MAP_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except self.anthropic.APIError as e:
            log("analyze", f"window {idx} failed: {e}")
            return "(this segment could not be analyzed)"


class Synthesizer:
    """Reduce step: combine all window observations into one understanding."""

    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5)

    def synthesize(self, meta: dict, observations: list[dict],
                   full_transcript: str) -> str:
        def obs_line(o: dict) -> str:
            text = o["text"]
            # Keep every segment visible (so the model never assumes continuous
            # coverage) but render non-substantive ones as a neutral gap rather
            # than feeding a raw error string into the prompt.
            if (text.startswith("(this segment could not be analyzed")
                    or text.startswith("(no frames available")):
                return (f"[{format_ts(o['start'])} to {format_ts(o['end'])}] "
                        f"(no observation available for this segment)")
            return f"[{format_ts(o['start'])} to {format_ts(o['end'])}] {text}"

        obs_block = "\n\n".join(obs_line(o) for o in observations)
        transcript_block = full_transcript.strip() or "(no speech was detected)"

        prompt = (
            f"Video file: {meta.get('video_name')}\n"
            f"Approximate length analyzed: {format_ts(meta.get('analyzed_seconds', 0))}\n\n"
            f"=== TIME-ORDERED OBSERVATIONS OF EACH SEGMENT ===\n{obs_block}\n\n"
            f"=== FULL TRANSCRIPT ===\n{transcript_block}\n\n"
            f"Using only the material above, write the understanding of this video "
            f"in Markdown with these sections:\n"
            f"## Overview\n"
            f"A short paragraph capturing what the video is and what it is about.\n"
            f"## Timeline\n"
            f"A bulleted, timestamped walk through the main beats.\n"
            f"## What is shown on screen\n"
            f"Key visuals, settings, people or objects, and any on-screen text or graphics.\n"
            f"## What is said\n"
            f"The main points, claims, or messages from the spoken content.\n"
            f"## Purpose and takeaway\n"
            f"Who this seems to be for and what a viewer is meant to come away with.\n"
        )

        if meta.get("config", {}).get("social"):
            # Folded into this same reduce call: no extra API request, just more
            # of the output the creator asked for, grounded in the material above.
            prompt += (
                "## Hook options\n"
                "Three short, scroll-stopping opening lines (max 12 words each) a "
                "creator could say or caption in the first 2 seconds, grounded in "
                "the actual content above. No clickbait the video does not deliver.\n"
                "## Caption\n"
                "One ready-to-post caption of 1 to 3 sentences in a natural creator "
                "voice, plus a single clear call to action.\n"
                "## Hashtags\n"
                "A single space-separated line of 10 to 15 lowercase hashtags "
                "relevant to the actual topic, ordered broad to niche. Only tags "
                "the content supports.\n"
                "## TL;DR\n"
                "One sentence a viewer could read to decide whether to watch.\n"
                "## Repurpose ideas\n"
                "Three short-form clip ideas, each with the timestamp range to cut, "
                "drawn from the timeline above.\n"
            )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REDUCE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except self.anthropic.APIError as e:
            # Do not throw away every paid map call: write a usable deliverable
            # from the observations we already captured.
            log("synthesize", f"synthesis call failed: {e}")
            return (
                f"# Understanding (synthesis failed)\n\n"
                f"The final synthesis call failed: {e}\n\n"
                f"The per-window observations below were captured successfully "
                f"and are also saved in observations.json. You can re-run "
                f"synthesis from them without repeating the vision calls.\n\n"
                f"## Time-ordered observations\n\n{obs_block}\n\n"
                f"## Full transcript\n\n{transcript_block}\n"
            )


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

class VideoWatcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.work = cfg.out_dir
        self.frames_dir = self.work / "frames"
        self.time_offset = 0.0

    def _prepare_source(self) -> tuple[Path, float, float]:
        """
        Returns (source_path, time_offset, analyzed_duration).
        If start/end are set, makes a trimmed working copy and reports the
        offset so all timestamps map back onto the original video timeline.

        Uses input-side seek (-ss before -i) and re-encodes the trim so both
        the video and audio streams restart at t=0 and the reported offset
        (== start) holds exactly. A stream copy would leave the video PTS
        unzeroed, knocking frame timestamps out of sync with the transcript.
        """
        src = self.cfg.video
        if self.cfg.start is None and self.cfg.end is None:
            return src, 0.0, probe_duration(src)

        trimmed = self.work / "_trimmed.mp4"
        cmd = ["ffmpeg", "-y"]
        if self.cfg.start is not None:
            cmd += ["-ss", str(self.cfg.start)]   # before -i => accurate input seek
        cmd += ["-i", str(src)]
        if self.cfg.end is not None:
            dur = self.cfg.end - (self.cfg.start or 0.0)
            cmd += ["-t", str(dur)]
        cmd += [str(trimmed)]                      # no -c copy: re-encode for an exact cut
        log("setup", f"trimming to [{self.cfg.start}, {self.cfg.end}] seconds")
        run(cmd)
        offset = float(self.cfg.start or 0.0)
        return trimmed, offset, probe_duration(trimmed)

    def _build_windows(self, start_t: float, end_t: float) -> list[tuple[float, float]]:
        windows = []
        n = max(1, math.ceil((end_t - start_t) / self.cfg.window))
        for i in range(n):
            ws = start_t + i * self.cfg.window
            we = min(start_t + (i + 1) * self.cfg.window, end_t)
            if we > ws:
                windows.append((ws, we))
        return windows

    @staticmethod
    def _frames_in_window(frames, ws, we, cap, is_last=False):
        # The final window drops its upper bound so a trailing frame whose pts
        # rounds up to exactly end_t is still analyzed instead of silently lost.
        if is_last:
            sel = [(t, p) for (t, p) in frames if ws <= t]
        else:
            sel = [(t, p) for (t, p) in frames if ws <= t < we]
        if len(sel) > cap:
            step = len(sel) / cap
            idx = sorted({int(i * step) for i in range(cap)})
            sel = [sel[i] for i in idx if i < len(sel)]
        return sel

    @staticmethod
    def _transcript_in_window(segments, ws, we) -> str:
        parts = [s["text"] for s in segments if s["start"] < we and s["end"] > ws]
        return " ".join(parts)

    def _load_resumable(self, windows) -> dict:
        """Reload a prior observations.json and return {index: record} for
        windows that completed successfully and still match the current
        windowing. Failure placeholders and empty windows are never reused."""
        path = self.work / "observations.json"
        if not path.exists():
            return {}
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(prior, list):
            return {}
        by_index: dict[int, dict] = {}
        for rec in prior:
            if not isinstance(rec, dict):
                continue
            idx = rec.get("index")
            text = rec.get("text", "")
            if not isinstance(idx, int) or not (1 <= idx <= len(windows)):
                continue
            ws, we = windows[idx - 1]
            if abs(rec.get("start", -1) - round(ws, 2)) > 0.01:
                continue
            if abs(rec.get("end", -1) - round(we, 2)) > 0.01:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            if (text.startswith("(this segment could not be analyzed")
                    or text.startswith("(no frames available")
                    or text.startswith("(analysis failed")):
                continue
            by_index[idx] = rec
        return by_index

    def run(self) -> None:
        cfg = self.cfg
        self.work.mkdir(parents=True, exist_ok=True)

        # --- setup / optional trim ----------------------------------------
        source, offset, analyzed = self._prepare_source()
        self.time_offset = offset
        start_t, end_t = offset, offset + analyzed
        log("setup", f"analyzing {format_ts(analyzed)} of video "
                     f"(timeline {format_ts(start_t)} to {format_ts(end_t)})")

        # --- stage 1: listen ----------------------------------------------
        segments: list[dict] = []
        if not cfg.no_audio:
            audio = self.work / "audio.wav"
            log("listen", "extracting audio track")
            extract_audio(source, audio)
            transcriber = Transcriber(
                cfg.whisper_model, cfg.device, cfg.compute_type, cfg.language
            )
            segments = transcriber.transcribe(audio, time_offset=offset)
        else:
            log("listen", "skipped (--no-audio)")

        full_transcript = "\n".join(
            f"[{format_ts(s['start'])}] {s['text']}" for s in segments
        )

        # --- stage 2: see -------------------------------------------------
        extractor = FrameExtractor(self.frames_dir, cfg.frame_width, cfg.jpeg_q)
        frames = extractor.extract(source, cfg, analyzed, time_offset=offset)
        if not frames:
            die("No frames could be extracted from the video.")

        # --- plan windows -------------------------------------------------
        windows = self._build_windows(start_t, end_t)
        if not windows:
            die("Nothing to analyze: the video has zero usable duration "
                "(check the file, or your --start/--end range).")
        total_calls = len(windows)
        log("plan", f"{len(frames)} frames across {len(windows)} windows "
                    f"of {cfg.window:g}s")
        log("plan", f"this will make up to {total_calls} vision calls "
                    f"({cfg.map_model}, {cfg.concurrency} at a time) "
                    f"+ 1 synthesis call ({cfg.reduce_model})")

        # Rough, pre-run cost estimate (images dominate; figures are approximate).
        est_imgs = min(len(frames), len(windows) * cfg.max_frames_per_window)
        map_in = est_imgs * 600 + len(windows) * 500
        map_out = len(windows) * 250
        red_in = len(windows) * 120 + len(full_transcript) // 4
        red_out = (5000 if cfg.social else cfg.reduce_max_tokens) // 2
        mp = _PRICING.get(cfg.map_model, (3.0, 15.0))
        rd = _PRICING.get(cfg.reduce_model, (3.0, 15.0))
        est_cost = (map_in * mp[0] + map_out * mp[1]
                    + red_in * rd[0] + red_out * rd[1]) / 1_000_000
        log("plan", f"rough cost estimate ~${est_cost:.2f} "
                    f"(~{est_imgs} images sent; very approximate)")

        # --- write what we have so far ------------------------------------
        meta = {
            "video_name": cfg.video.name,
            "video_path": str(cfg.video.resolve()),
            "source_url": cfg.source_url,
            "analyzed_seconds": round(analyzed, 2),
            "timeline_start": round(start_t, 2),
            "timeline_end": round(end_t, 2),
            "n_frames": len(frames),
            "n_windows": len(windows),
            "frame_timestamps": [round(t, 2) for (t, _p) in frames],
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()},
        }
        (self.work / "transcript.txt").write_text(full_transcript, encoding="utf-8")
        (self.work / "transcript.json").write_text(
            json.dumps(segments, indent=2), encoding="utf-8"
        )
        (self.work / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if cfg.dry_run:
            log("done", "dry run complete. transcript and frames are ready, "
                        "no API calls were made.")
            log("done", f"output folder: {self.work}")
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            die("ANTHROPIC_API_KEY is not set. Set it, or re-run with --dry-run.")

        # --- stage 3: analyze (map), windows run concurrently -------------
        analyzer = VisionAnalyzer(cfg.map_model, cfg.map_max_tokens)
        n = len(windows)
        results: list[Optional[dict]] = [None] * n
        lock = threading.Lock()
        done = 0

        reused: dict[int, dict] = {}
        if cfg.resume:
            reused = self._load_resumable(windows)
            if reused:
                log("analyze", f"resuming: reusing {len(reused)} previously "
                               f"completed window(s)")

        def write_prefix() -> None:
            # Write the longest in-order, fully-completed prefix, so a crash
            # always leaves observations.json as a valid prefix (never sparse).
            prefix: list[dict] = []
            for r in results:
                if r is None:
                    break
                prefix.append(r)
            (self.work / "observations.json").write_text(
                json.dumps(prefix, indent=2), encoding="utf-8"
            )

        def work(i0: int, ws: float, we: float) -> None:
            nonlocal done
            if (i0 + 1) in reused:
                rec = reused[i0 + 1]
            else:
                wf = self._frames_in_window(
                    frames, ws, we, cfg.max_frames_per_window,
                    is_last=(i0 + 1 == n),
                )
                wt = self._transcript_in_window(segments, ws, we)
                if not wf:
                    text = "(no frames available for this window)"
                else:
                    text = analyzer.analyze_window(i0 + 1, ws, we, wf, wt)
                rec = {"index": i0 + 1, "start": round(ws, 2), "end": round(we, 2),
                       "n_frames": len(wf), "text": text}
            with lock:
                results[i0] = rec
                done += 1
                log("analyze", f"{done}/{n} done "
                               f"[{format_ts(ws)}-{format_ts(we)}] "
                               f"{rec['n_frames']} frames")
                write_prefix()

        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
            futs = [ex.submit(work, i, ws, we)
                    for i, (ws, we) in enumerate(windows)]
            for f in futs:
                f.result()  # surface any unexpected (non-APIError) exception

        observations = [r for r in results if r is not None]
        (self.work / "observations.json").write_text(
            json.dumps(observations, indent=2), encoding="utf-8"
        )

        # --- stage 4: synthesize (reduce) ---------------------------------
        log("synthesize", "combining observations into the final understanding")
        # The social sections add length; raise the cap (not a charge, output
        # bills only for tokens generated) so they are not truncated.
        reduce_tokens = (max(cfg.reduce_max_tokens, 5000)
                         if cfg.social else cfg.reduce_max_tokens)
        synth = Synthesizer(cfg.reduce_model, reduce_tokens)
        understanding = synth.synthesize(meta, observations, full_transcript)
        (self.work / "understanding.md").write_text(understanding, encoding="utf-8")

        # --- report -------------------------------------------------------
        print("\n" + "=" * 70)
        print(understanding)
        print("=" * 70 + "\n")
        log("done", f"output folder: {self.work}")
        log("done", "files: understanding.md, transcript.txt, transcript.json, "
                    "observations.json, meta.json, frames/")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch and understand a video with Claude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Kept as a raw string (not type=Path) so an http(s) URL is not mangled into
    # a Windows path; it is turned into a Path or downloaded in main().
    p.add_argument("video",
                   help="local video file path, or a TikTok/YouTube/Instagram "
                        "URL to download with yt-dlp")
    p.add_argument("--out", type=Path, default=None,
                   help="output folder (default: ./<video name>_watch)")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true",
                   help="quick, cheap skim preset (tiny whisper, fewer frames)")
    g.add_argument("--thorough", action="store_true",
                   help="higher-fidelity preset (finer windows, Sonnet map model)")

    p.add_argument("--start", type=float, default=None,
                   help="analyze only from this many seconds in")
    p.add_argument("--end", type=float, default=None,
                   help="analyze only up to this many seconds")

    p.add_argument("--no-audio", action="store_true",
                   help="skip transcription and use vision only")
    p.add_argument("--whisper-model", default="base",
                   help="faster-whisper size: tiny, base, small, medium, large-v3")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="device for whisper")
    p.add_argument("--compute-type", default="int8",
                   help="whisper compute type, e.g. int8 (cpu) or float16 (gpu)")
    p.add_argument("--language", default=None,
                   help="force a language code (e.g. en), default auto detect")

    p.add_argument("--no-scene", action="store_true",
                   help="disable change detection and sample on a fixed grid only")
    p.add_argument("--scene-threshold", type=float, default=0.20,
                   help="change sensitivity, 0..1, lower catches subtler motion")
    p.add_argument("--fps-interval", type=float, default=3.0,
                   help="fill cadence used only to bridge long static gaps (0 disables)")
    p.add_argument("--max-gap", type=float, default=6.0,
                   help="never leave a stretch longer than this many seconds unsampled")
    p.add_argument("--min-gap", type=float, default=1.5,
                   help="do not keep frames closer together than this many seconds")
    p.add_argument("--max-frames", type=int, default=120,
                   help="overall cap on extracted frames")
    p.add_argument("--frame-width", type=int, default=768,
                   help="downscale the long edge of each frame to this many px")
    p.add_argument("--jpeg-q", type=int, default=4,
                   help="ffmpeg jpeg quality, 2 best to 31 worst")

    p.add_argument("--window", type=float, default=30.0,
                   help="seconds of video analyzed per vision call")
    p.add_argument("--max-frames-per-window", type=int, default=10,
                   help="cap on frames sent in a single vision call")
    p.add_argument("--concurrency", type=int, default=4,
                   help="number of vision (map) calls to run in parallel")
    p.add_argument("--map-model", default="claude-haiku-4-5-20251001",
                   help="model for per-window vision analysis")
    p.add_argument("--reduce-model", default="claude-sonnet-4-6",
                   help="model for the final synthesis")

    p.add_argument("--dry-run", action="store_true",
                   help="extract frames and transcribe, but make no API calls")
    p.add_argument("--resume", action="store_true",
                   help="reuse completed window observations from a previous run "
                        "and skip re-analyzing (and re-paying for) them")
    p.add_argument("--social", action="store_true",
                   help="also emit creator hook, caption, hashtags, TL;DR, and "
                        "repurpose ideas in understanding.md")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Detect which optional flags the user actually passed (via a SUPPRESS-default
    # twin parser) so a preset only fills gaps and never overrides an explicit one.
    sentinel = build_parser()
    for a in sentinel._actions:
        if a.dest != "help":
            a.default = argparse.SUPPRESS
    supplied = set(vars(sentinel.parse_args(
        argv if argv is not None else sys.argv[1:])).keys())

    # --fast / --thorough bundle existing, tested flags. --fast sets max_gap=12 so
    # fps_interval=8 survives the soft-clamp below instead of being lowered to 6.
    FAST = {"window": 45.0, "fps_interval": 8.0, "max_gap": 12.0,
            "max_frames": 60, "max_frames_per_window": 6,
            "whisper_model": "tiny", "concurrency": 8}
    THOROUGH = {"window": 20.0, "fps_interval": 2.0, "scene_threshold": 0.15,
                "max_frames": 160, "max_frames_per_window": 10,
                "whisper_model": "small", "map_model": "claude-sonnet-4-6"}
    preset = FAST if args.fast else THOROUGH if args.thorough else None
    if preset:
        for dest, val in preset.items():
            if dest not in supplied:
                setattr(args, dest, val)
        log("setup", f"preset --{'fast' if args.fast else 'thorough'} applied "
                     f"(your explicit flags still win)")

    check_binaries()

    # The positional may be a local path or an http(s) URL.
    url_mode = is_url(args.video)
    if not url_mode and not Path(args.video).exists():
        die(f"video not found: {args.video}")

    # Fail fast on operator errors, before minutes of transcription/extraction.
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        die("ANTHROPIC_API_KEY is not set. Set it, or re-run with --dry-run.\n"
            "  macOS/Linux:        export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Windows PowerShell: $env:ANTHROPIC_API_KEY=\"sk-ant-...\"\n"
            "  Windows cmd:        set ANTHROPIC_API_KEY=sk-ant-...")

    if args.max_frames < 1:
        die("--max-frames must be a positive integer")
    if args.max_frames_per_window < 1:
        die("--max-frames-per-window must be a positive integer")

    if args.start is not None and args.start < 0:
        die("--start must be >= 0")
    if args.end is not None and args.end < 0:
        die("--end must be >= 0")
    if (args.start is not None and args.end is not None
            and args.end <= args.start):
        die(f"--end ({args.end}) must be greater than --start ({args.start})")
    if not url_mode and args.start is not None:
        total = probe_duration(Path(args.video))
        if args.start >= total:
            die(f"--start ({args.start}s) is at or past the end of the "
                f"video ({total:.2f}s)")

    # Resolve the input: download a URL with yt-dlp, or use the local path as-is.
    source_url = None
    if url_mode:
        source_url = args.video
        out_dir = args.out or Path("video_watch")
        video_path = download_url(source_url, Path(out_dir) / "source")
    else:
        video_path = Path(args.video)
        out_dir = args.out or (video_path.parent / f"{video_path.stem}_watch")

    cfg = Config(
        video=video_path,
        out_dir=out_dir,
        start=args.start,
        end=args.end,
        no_audio=args.no_audio,
        whisper_model=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        use_scene=not args.no_scene,
        scene_threshold=args.scene_threshold,
        fps_interval=args.fps_interval,
        max_gap=args.max_gap,
        min_gap=args.min_gap,
        max_frames=args.max_frames,
        frame_width=args.frame_width,
        jpeg_q=args.jpeg_q,
        window=args.window,
        max_frames_per_window=args.max_frames_per_window,
        concurrency=args.concurrency,
        map_model=args.map_model,
        reduce_model=args.reduce_model,
        dry_run=args.dry_run,
        resume=args.resume,
        social=args.social,
        source_url=source_url,
    )

    # Soft-clamp the frame-spacing knobs so the min_gap thinning pass cannot
    # silently undo the bridging pass and re-open long unsampled stretches.
    if cfg.fps_interval and cfg.fps_interval > 0:
        if cfg.fps_interval > cfg.max_gap:
            log("see", f"fps-interval {cfg.fps_interval:g} is larger than "
                       f"max-gap {cfg.max_gap:g}; lowering fps-interval to "
                       f"{cfg.max_gap:g}")
            cfg.fps_interval = cfg.max_gap
        if cfg.min_gap > cfg.fps_interval:
            log("see", f"min-gap {cfg.min_gap:g} is larger than fps-interval "
                       f"{cfg.fps_interval:g}; capping min-gap to "
                       f"{cfg.fps_interval:g}")
            cfg.min_gap = cfg.fps_interval

    try:
        VideoWatcher(cfg).run()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        die(f"a command failed:\n{' '.join(e.cmd)}\n\n{stderr[-1500:]}")
    except KeyboardInterrupt:
        die("interrupted by user", code=130)


if __name__ == "__main__":
    main()
