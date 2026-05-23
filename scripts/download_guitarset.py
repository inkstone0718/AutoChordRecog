# -*- coding: utf-8 -*-
"""
download_guitarset.py - Automation script to download and prepare the GuitarSet dataset.
Downloads audio (mono-mic) and JAMS annotations from Zenodo, converts the JAMS
annotations to .lab files, and organizes the directories under data/audio and data/labels.
"""

import os
import zipfile
import json
import urllib.request
import shutil

# URLs for GuitarSet on Zenodo
ANNOTATION_URL = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"
AUDIO_URL = "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1"

# Target directories
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
TEMP_DIR = "temp_guitarset"

def download_file(url, output_path):
    """
    Downloads a file with basic download progress printout.
    """
    print(f"Downloading {url} \nTo: {output_path}...")
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            s = f"\rProgress: {percent:5.1f}% [{read_so_far / 1024**2:5.1f} MB / {total_size / 1024**2:5.1f} MB]"
            print(s, end="")
        else:
            print(f"\rRead {read_so_far / 1024**2:.1f} MB", end="")
            
    urllib.request.urlretrieve(url, output_path, reporthook=report_hook)
    print("\nDownload complete!")

def convert_jams_to_lab(jams_path, lab_output_path):
    """
    Parses a JAMS annotation file and writes chord intervals in .lab format.
    Format: start_time end_time chord
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        jams_data = json.load(f)
        
    intervals = []
    # Search for chord annotation namespace
    for ann in jams_data.get("annotations", []):
        if ann.get("namespace") == "chord":
            for obs in ann.get("data", []):
                start = obs.get("time")
                duration = obs.get("duration")
                end = start + duration
                chord = obs.get("value")
                intervals.append((start, end, chord))
            break # Usually one chord annotation per file
            
    # Write to lab file
    with open(lab_output_path, "w", encoding="utf-8") as f:
        for start, end, chord in intervals:
            f.write(f"{start:.3f}\t{end:.3f}\t{chord}\n")

def main():
    # Make sure target directories exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    annotation_zip = os.path.join(TEMP_DIR, "annotation.zip")
    audio_zip = os.path.join(TEMP_DIR, "audio_mono-mic.zip")
    
    # 1. Download and process annotations (small file, ~1.5 MB)
    print("\n=== Processing Annotations ===")
    if not os.path.exists(annotation_zip):
        download_file(ANNOTATION_URL, annotation_zip)
    else:
        print("annotation.zip already downloaded.")
        
    print("Extracting annotations...")
    temp_ann_extract = os.path.join(TEMP_DIR, "annotation_extracted")
    with zipfile.ZipFile(annotation_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_ann_extract)
        
    # Convert JAMS to .lab files
    jams_files = []
    for root, _, files in os.walk(temp_ann_extract):
        for file in files:
            if file.endswith(".jams"):
                jams_files.append(os.path.join(root, file))
                
    print(f"Found {len(jams_files)} JAMS files. Converting to .lab files...")
    for jf in jams_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        # Remove any suffix that indicates annotation type (e.g. annotation_extracted/xxx.jams)
        lab_name = f"{base_name}.lab"
        lab_path = os.path.join(LABEL_DIR, lab_name)
        convert_jams_to_lab(jf, lab_path)
        
    print(f"Completed! Saved {len(jams_files)} .lab files in '{LABEL_DIR}'.")
    
    # 2. Download and process audio (large file, ~1 GB)
    print("\n=== Processing Audio ===")
    print("WARNING: The audio zip file is approx 1.0 GB. This may take several minutes to download.")
    
    # Check if user wants to download the audio now
    confirm = input("Do you want to proceed with downloading the audio? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Skipping audio download. You can download audio_mono-mic.zip manually and place it in the temp_guitarset/ directory.")
        return
        
    if not os.path.exists(audio_zip):
        download_file(AUDIO_URL, audio_zip)
    else:
        print("audio_mono-mic.zip already downloaded.")
        
    print("Extracting audio files...")
    temp_audio_extract = os.path.join(TEMP_DIR, "audio_extracted")
    with zipfile.ZipFile(audio_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_audio_extract)
        
    # Move audio files (.wav) to data/audio/
    moved_count = 0
    for root, _, files in os.walk(temp_audio_extract):
        for file in files:
            if file.endswith(".wav"):
                src_path = os.path.join(root, file)
                # GuitarSet audio has subdirectories; we flatten them into data/audio
                dest_path = os.path.join(AUDIO_DIR, file)
                shutil.move(src_path, dest_path)
                moved_count += 1
                
    print(f"Completed! Moved {moved_count} audio files to '{AUDIO_DIR}'.")
    
    # Clean up temp directory
    print("\nCleaning up temporary files...")
    try:
        shutil.rmtree(TEMP_DIR)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Could not delete temp directory: {e}")
        
    print("\n=== GuitarSet Setup Finished Successfully! ===")
    print("You can now run 'python src/train.py' to train on the GuitarSet dataset.")

if __name__ == '__main__':
    main()
