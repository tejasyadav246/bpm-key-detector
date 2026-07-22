import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.set_page_config(page_title="BPM & Key Detector", page_icon="🎵")

st.title("🎵 Track BPM & Key Detector")
st.write("Upload a WAV or MP3 audio file to analyze its tempo and musical key.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

def get_krumhansl_profiles():
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.28])
    major_profile = (major_profile - np.mean(major_profile)) / np.std(major_profile)
    minor_profile = (minor_profile - np.mean(minor_profile)) / np.std(minor_profile)
    return major_profile, minor_profile

def detect_key(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = (chroma_mean - np.mean(chroma_mean)) / np.std(chroma_mean)
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_prof, minor_prof = get_krumhansl_profiles()
    results = []

    for i in range(12):
        maj_shifted = np.roll(major_prof, i)
        min_shifted = np.roll(minor_prof, i)
        r_major = np.corrcoef(chroma_mean, maj_shifted)[0, 1]
        r_minor = np.corrcoef(chroma_mean, min_shifted)[0, 1]
        results.append((pitch_names[i] + " Major", r_major))
        results.append((pitch_names[i] + " Minor", r_minor))

    results.sort(key=lambda x: x[1], reverse=True)
    top_key, score = results[0]
    confidence = max(0.0, min(1.0, (score + 1) / 2))
    return top_key, confidence

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("Analyzing audio signal..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            _, y_percussive = librosa.effects.hpss(y)
            tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]

            key, confidence = detect_key(y, sr)

            st.success("Analysis Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{duration:.1f} s")
            col2.metric("Tempo", f"{tempo:.1f} BPM")
            col3.metric("Key", key, f"{confidence*100:.0f}% confidence")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
