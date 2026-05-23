# -*- coding: utf-8 -*-
"""
preprocess.py - Audio processing and label alignment pipeline.
Loads audio and annotation files, extracts chroma CQT features,
and slices them into windowed frames ready for CNN training.
"""

import librosa
import numpy as np

try:
    from src.utils import simplify_chord, CHORD_TO_IDX
except ImportError:
    from utils import simplify_chord, CHORD_TO_IDX

# Standard configuration constants
SR = 22050          # Audio sample rate
HOP_LENGTH = 2048   # Hop length (~93ms frames at 22050Hz for efficient temporal context)
WINDOW_SIZE = 11    # Convolution window size (5 frames before, 1 center, 5 after)

def load_label_intervals(label_path):
    """
    Parses a tab/space-delimited annotation file (.lab or .txt)
    Format: start_time end_time chord_name
    """
    intervals = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    chord = parts[2]
                    intervals.append((start, end, chord))
                except ValueError:
                    # Skip header or malformed lines
                    continue
    return intervals

def align_labels_to_frames(frame_times, intervals):
    """
    Assigns a chord label to each frame's center time based on annotation intervals.
    """
    labels = []
    for t in frame_times:
        matched_chord = 'N'
        # Search for interval containing the current timestamp
        for start, end, chord in intervals:
            if start <= t <= end:
                matched_chord = chord
                break
        
        # Simplify the chord and map to index
        simplified = simplify_chord(matched_chord)
        label_idx = CHORD_TO_IDX.get(simplified, CHORD_TO_IDX['N'])
        labels.append(label_idx)
        
    return np.array(labels)

def extract_chroma_windows(audio_path, window_size=WINDOW_SIZE):
    """
    Extracts CQT chroma features from audio and groups them into sliding windows.
    Returns:
        features: numpy array of shape (num_frames, 12, window_size, 1)
        frame_times: timestamp for the center of each frame
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SR)
    
    # Extract CQT Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    num_frames = chroma.shape[1]
    
    # Calculate frame timestamps
    frame_times = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=HOP_LENGTH)
    
    # Pad chroma features temporally to handle boundary frames
    half_window = window_size // 2
    chroma_padded = np.pad(chroma, ((0, 0), (half_window, half_window)), mode='constant')
    
    # Slice sliding windows
    features = []
    for i in range(num_frames):
        window = chroma_padded[:, i : i + window_size]
        features.append(window)
        
    features = np.array(features)                       # Shape: (num_frames, 12, window_size)
    features = np.expand_dims(features, axis=-1)        # Shape: (num_frames, 12, window_size, 1)
    
    return features, frame_times

def preprocess_track(audio_path, label_path=None, window_size=WINDOW_SIZE):
    """
    Processes a single audio file and matches labels if a label path is provided.
    """
    features, frame_times = extract_chroma_windows(audio_path, window_size)
    
    if label_path is not None:
        intervals = load_label_intervals(label_path)
        labels = align_labels_to_frames(frame_times, intervals)
        return features, labels
    
    return features, frame_times

if __name__ == '__main__':
    # Test preprocessing with a dummy label file
    import tempfile
    import os
    
    print("Testing preprocessing workflow...")
    
    # 1. Create a dummy label file
    dummy_label_content = "0.00 2.50 C\n2.50 5.00 G\n5.00 7.50 Am\n7.50 10.00 F\n"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.lab') as tmp:
        tmp.write(dummy_label_content)
        tmp_label_path = tmp.name
        
    # 2. Get librosa built-in trumpet file for testing
    dummy_audio_path = librosa.ex('trumpet')
    
    try:
        features, labels = preprocess_track(dummy_audio_path, tmp_label_path)
        print(f"Preprocessed features shape: {features.shape}")
        print(f"Preprocessed labels shape: {labels.shape}")
        print(f"Unique label indices present: {np.unique(labels)}")
        print("Preprocessing test completed successfully!")
    finally:
        os.remove(tmp_label_path)
