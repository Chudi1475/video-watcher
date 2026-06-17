# Video Watcher

A command line tool that "watches" a video the way an attentive person would. It
listens to the audio, looks at the picture across time, and produces a single
faithful account of what the video is and what happens in it.

It does this in four stages:

1. **Listen.** Pull the audio and transcribe speech with faster-whisper, keeping
   per segment timestamps.
2. **See.** Extract still frames with ffmpeg using adaptive sampling: densely
   wherever the picture is changing (cuts and motion, the moments a viewer
   notices), and periodically across static stretches so nothing is left
   unsampled for too long. Then de-duplicate and cap the count.
3. **Analyze.** Slice the video into short windows. For each window, send the
   frames in that window plus the words spoken during it to Claude's vision
   model and ask what is happening. This is the map step.
4. **Synthesize.** Feed every window description plus the full transcript back to
   Claude for one combined understanding. This is the reduce step.

## Why this design

A single image tells Claude what a scene looks like, but not what came before it
or what is being said over it. By pairing time stamped frames with the matching
transcript and walking the video window by window, the tool reconstructs the
viewing experience: sequence, on screen text, actions, and how the visuals line
up with the narration. The final pass turns those notes into one coherent recap.

## Requirements

- Python 3.9 or newer
- ffmpeg and ffprobe on your PATH
  - Ubuntu or Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg` / `scoop install ffmpeg`)
- `pip install -r requirements.txt`
- An Anthropic API key in your environment:
  - macOS or Linux: `export ANTHROPIC_API_KEY=sk-ant-...`
  - Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`
  - Windows cmd: `set ANTHROPIC_API_KEY=sk-ant-...`

The first run downloads the chosen Whisper model once and caches it.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# watch a video end to end
python video_watcher.py myvideo.mp4
```

When it finishes, look in the new folder `myvideo_watch/`:

- `understanding.md`  the final account of the video (the main output)
- `transcript.txt`    full transcript with timestamps
- `transcript.json`   raw transcript segments
- `observations.json` per window descriptions
- `meta.json`         run parameters and the kept frame timestamps
- `frames/`           the extracted JPEG frames

## Useful options

```bash
# free, no API calls: just transcribe and pull frames so you can inspect them
python video_watcher.py myvideo.mp4 --dry-run

# only analyze a slice (seconds), great for testing on long videos
python video_watcher.py long.mp4 --start 60 --end 180

# more thorough: more frames, finer windows, stronger map model
python video_watcher.py myvideo.mp4 --window 20 --fps-interval 2 \
  --map-model claude-sonnet-4-6

# cheaper and faster: fewer frames, larger windows
python video_watcher.py myvideo.mp4 --window 45 --fps-interval 8 --max-frames 60

# one-flag presets: --fast (quick skim) or --thorough (higher fidelity)
python video_watcher.py myvideo.mp4 --fast
python video_watcher.py myvideo.mp4 --thorough
# presets only fill in flags you did not set, so your explicit flags always win
python video_watcher.py myvideo.mp4 --fast --window 10

# creator extras: hook, ready-to-post caption, hashtags, clip ideas (free, same call)
python video_watcher.py myvideo.mp4 --social

# faster on long videos: run more vision calls at once
python video_watcher.py myvideo.mp4 --concurrency 8

# pick up where a crashed or interrupted run left off (skips re-paid calls)
python video_watcher.py myvideo.mp4 --resume

# vision only, no speech (silent clips or B roll)
python video_watcher.py myvideo.mp4 --no-audio

# better transcription quality (slower on CPU)
python video_watcher.py myvideo.mp4 --whisper-model small

# on a CUDA GPU
python video_watcher.py myvideo.mp4 --device cuda --compute-type float16
```

Run `python video_watcher.py --help` for the full list.

### Key flags

| Flag | What it does | Default |
| --- | --- | --- |
| `--window` | seconds of video per vision call | 30 |
| `--scene-threshold` | change sensitivity, 0 to 1, lower catches subtler motion | 0.20 |
| `--fps-interval` | fill cadence to bridge long static gaps (0 disables) | 3.0 |
| `--max-gap` | never leave a stretch longer than this unsampled | 6.0 |
| `--min-gap` | do not keep frames closer together than this | 1.5 |
| `--max-frames` | overall cap on extracted frames | 120 |
| `--max-frames-per-window` | cap on frames sent in one vision call | 10 |
| `--frame-width` | downscale the long edge of each frame | 768 |
| `--concurrency` | vision (map) calls to run in parallel | 4 |
| `--fast` | quick, cheap skim preset (tiny whisper, fewer frames) | off |
| `--thorough` | higher-fidelity preset (finer windows, Sonnet map) | off |
| `--social` | also emit creator hook, caption, hashtags, clip ideas | off |
| `--map-model` | model for per window analysis | claude-haiku-4-5-20251001 |
| `--reduce-model` | model for the final synthesis | claude-sonnet-4-6 |
| `--dry-run` | extract and transcribe, make no API calls | off |
| `--resume` | reuse completed observations from a prior run | off |

## Watch a video by URL (TikTok / YouTube / Instagram)

Pass a link instead of a file path and it downloads automatically with
[yt-dlp](https://github.com/yt-dlp/yt-dlp) (a free, optional dependency), then
runs the normal pipeline.

```bash
pip install -U yt-dlp   # one time, only needed for URLs

python video_watcher.py "https://www.tiktok.com/@user/video/123456" --social
python video_watcher.py "https://youtu.be/VIDEO_ID" --fast
python video_watcher.py "https://www.instagram.com/reel/CODE/" --social
```

The video is saved into `./video_watch/source/` and all outputs land in
`./video_watch/`. Local files work exactly as before; the download step only
runs when the argument is an `http(s)` URL.

Notes:
- For private or age-gated videos, yt-dlp can use your browser login:
  add `--cookies-from-browser chrome` (or `firefox`, `edge`).
- Instagram public links sometimes need a retry. Sites change often, so run
  `pip install -U yt-dlp` if one stops working.

## Run it from your phone (Google Colab, free)

No install on the phone. Run everything in a free Google Colab notebook from
your phone's browser:

1. Push this repo to GitHub (already done if you cloned it from there), and in
   `video_watcher_colab.ipynb` set your GitHub username in the raw URL in Cell 2.
2. On your phone, open [colab.research.google.com](https://colab.research.google.com),
   sign in, then **File > Open notebook > GitHub** and open `video_watcher_colab.ipynb`.
3. Tap the **key icon** (Secrets) in the left sidebar, add a secret named
   `ANTHROPIC_API_KEY` with your `sk-ant-...` value, and enable notebook access.
4. Paste a TikTok / YouTube / Instagram link into the URL field in Cell 3.
5. **Runtime > Run all.** It installs everything, downloads the video, runs the
   tool (`--fast --social` by default, good for short clips on Colab's free CPU),
   and renders `understanding.md` inline.

For a big speed-up, switch the Colab runtime to a **free GPU** (Runtime > Change
runtime type > T4 GPU) and the notebook will run Whisper on it automatically.

## Cost and speed

The number of vision calls equals the number of windows, which is roughly the
analyzed length divided by `--window`. Each call sends up to
`--max-frames-per-window` small JPEGs. To keep things cheap and fast the defaults
use Haiku for the many per window calls and Sonnet for the single synthesis. Bump
the map model to Sonnet, or the reduce model to `claude-opus-4-8`, when you want
maximum fidelity. Frames are capped at 768 px on the long edge, which stays well
under the API limits for multi image requests and keeps each request small.

Run `--dry-run` first on anything long. It prints exactly how many calls a real
run would make before you spend anything.

## Using it inside Claude Code

1. Drop `video_watcher.py`, `requirements.txt`, and this README into a folder.
2. Open that folder in Claude Code.
3. Tell Claude Code: install the requirements, make sure ffmpeg is installed, set
   `ANTHROPIC_API_KEY`, then run `python video_watcher.py <path-to-video>
   --dry-run` and report the planned call count. After you approve, run it for
   real and open `understanding.md`.

Claude Code can also extend the tool from here. Good next steps are listed below.

## Limitations and honest extensions

- Whisper transcribes speech only. It does not label music, sound effects, or
  tone of voice. Long non speech spans show up as gaps in the transcript, which
  the synthesis can note as likely music or B roll. For real audio event tagging,
  add a model such as YAMNet or CLAP and merge its labels into each window.
- Frame sampling can still miss a very brief on screen moment between samples.
  Lower `--fps-interval` and `--scene-threshold` for dense, fast cut content.
- Trimming with `--start` and `--end` re-encodes the selected span so the cut is
  frame accurate and the frame and transcript timelines stay aligned. This is a
  little slower than a raw stream copy but avoids keyframe drift on the segment.
- Voice emotion and speaker identification are not included. Both are reasonable
  add ons (a diarization model for speakers, an audio emotion model for tone).
