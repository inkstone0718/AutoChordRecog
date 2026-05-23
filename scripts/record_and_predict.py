# -*- coding: utf-8 -*-
"""
record_and_predict.py - Interactive audio recording and chord transcription script.
Uses macOS default microphone via ffmpeg to record audio,
then runs inference on the recorded file to transcribe the chords.
"""

import os
import time
import subprocess
import sys

# Add the project root directory to the python path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_chords

DEFAULT_RECORDING_PATH = "my_recording.wav"
DEFAULT_MODEL_PATH = "best_chord_model.keras"

def record_audio(filename=DEFAULT_RECORDING_PATH):
    """
    Records audio from the default input device (microphone) using ffmpeg.
    Records until the user presses Enter.
    """
    print("\n=== Prepare to Record ===")
    print("Please get your guitar ready!")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
        
    print("\n🔴 RECORDING STARTED!")
    print("Play your chords clearly (e.g., C, G, Am, F).")
    print("👉 Press [ENTER] to stop recording...")
    
    # Run ffmpeg in background
    # "-f avfoundation -i :default" records from default macOS input
    cmd = [
        "ffmpeg",
        "-f", "avfoundation",
        "-i", ":default",
        "-y",
        filename
    ]
    
    # Start ffmpeg
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    try:
        # Wait for user to hit enter
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
        
    print("Stopping recording...")
    
    # Gracefully stop ffmpeg by sending 'q' to its stdin
    try:
        process.communicate(input=b'q', timeout=3)
    except (subprocess.TimeoutExpired, ValueError):
        process.terminate()
        process.wait()
        
    print(f"Recording saved to '{filename}'")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Record guitar chords and transcribe them.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained keras model.")
    parser.add_argument("--smoothing", type=int, default=7, help="Moving average smoothing window size.")
    parser.add_argument("--min-duration", type=float, default=0.4, help="Minimum duration of a chord segment.")
    parser.add_argument("--silence-threshold", type=float, default=0.015, help="RMS energy threshold for silence 'N'.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Warning: Trained model '{args.model}' not found.")
        print("Please train your model first on the GuitarSet dataset by running:")
        print("  python src/train.py")
        print("We can still record audio for you, but prediction will fail until trained.")
        
    record_audio(DEFAULT_RECORDING_PATH)
    
    if os.path.exists(args.model) and os.path.exists(DEFAULT_RECORDING_PATH):
        print("\n=== Transcribing Recorded Audio ===")
        predict_chords(
            DEFAULT_RECORDING_PATH, 
            model_path=args.model, 
            smoothing_window=args.smoothing, 
            min_duration=args.min_duration, 
            silence_threshold=args.silence_threshold
        )

if __name__ == '__main__':
    main()
