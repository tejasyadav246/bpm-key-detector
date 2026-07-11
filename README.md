# BPM & Key Detector

A Python command-line tool that analyzes an audio file and estimates its **tempo (BPM)** and **musical key** — the kind of info a producer manually figures out by ear before starting a remix or mashup.

## How it works

1. **Beat detection:** separates the percussive part of the audio (drums/transients) from the melodic part, since percussion gives a cleaner tempo signal. Runs it through `librosa`'s beat tracker to get BPM.
2. **Key detection:** builds a "chromagram" — how much energy the track has in each of the 12 notes (C through B) — and compares that pattern against the standard Krumhansl-Schmuckler key profiles for all 24 major/minor keys to find the closest match.

## Usage

```bash
pip install librosa numpy scipy
python detector.py path/to/song.wav
```

Example output:
```
--- Analysis Result ---
Duration : 8.0 sec
Tempo    : 117.5 BPM
Key      : C Major  (confidence: 0.85)
```

## Why I built this

As a producer, I manually work out BPM and key before sampling or remixing a track. This automates that first step and was also a way to apply signal-processing concepts from my AI/ML coursework to something I actually use.

## Possible extensions
- Batch-process an entire folder of tracks
- Export results to CSV for a DJ set-planning spreadsheet
- Add a simple GUI with drag-and-drop
