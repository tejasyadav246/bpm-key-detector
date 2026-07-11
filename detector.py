"""
BPM & Key Detector
-------------------
Analyzes an audio file and estimates its tempo (BPM) and musical key.

How it works (plain English):
1. Load the audio and separate out the beat-heavy "percussive" part —
   this makes tempo detection more accurate than using the raw mix.
2. Use librosa's beat tracker to estimate BPM from that percussive signal.
3. Compute a "chromagram" — this measures how much energy the song has
   in each of the 12 musical notes (C, C#, D... B) over time.
4. Compare the song's note-energy pattern against the known note patterns
   of all 24 major/minor keys (Krumhansl-Schmuckler key-profile method)
   and pick the closest match.

Usage:
    python detector.py path/to/song.wav
"""

import sys
import numpy as np
import librosa

# Standard Krumhansl-Schmuckler key profiles (relative strength of each
# note within a major or minor key, starting from the tonic).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_bpm(y, sr):
    """Estimate tempo in beats per minute."""
    y_percussive = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    tempo = np.atleast_1d(tempo)  # librosa may return a scalar or array
    return float(tempo[0])


def detect_key(y, sr):
    """Estimate musical key by correlating chroma energy with key profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)  # average note-energy across the song

    best_score = -np.inf
    best_key = None

    for shift in range(12):
        major_shifted = np.roll(MAJOR_PROFILE, shift)
        minor_shifted = np.roll(MINOR_PROFILE, shift)

        major_score = np.corrcoef(chroma_avg, major_shifted)[0, 1]
        minor_score = np.corrcoef(chroma_avg, minor_shifted)[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[shift]} Major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[shift]} Minor"

    return best_key, float(best_score)


def analyze(filepath):
    print(f"Loading: {filepath}")
    y, sr = librosa.load(filepath, sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    bpm = detect_bpm(y, sr)
    key, confidence = detect_key(y, sr)

    print("\n--- Analysis Result ---")
    print(f"Duration : {duration:.1f} sec")
    print(f"Tempo    : {bpm:.1f} BPM")
    print(f"Key      : {key}  (confidence: {confidence:.2f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detector.py <audio_file>")
        sys.exit(1)

    analyze(sys.argv[1])
