# -*- coding: utf-8 -*-
"""
train.py - Training pipeline for Automatic Chord Recognition.
Finds matched audio/label pairs, extracts features, trains the CNN model,
and saves the trained model.
"""

import os
import glob
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

try:
    from src.preprocess import preprocess_track, WINDOW_SIZE
    from src.model import build_chord_cnn
except ImportError:
    from preprocess import preprocess_track, WINDOW_SIZE
    from model import build_chord_cnn

# Constants
DEFAULT_AUDIO_DIR = os.path.join("data", "audio")
DEFAULT_LABEL_DIR = os.path.join("data", "labels")
MODEL_SAVE_PATH = "best_chord_model.keras"
HISTORY_SAVE_PATH = "training_history.json"

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def find_dataset_files(audio_dir=DEFAULT_AUDIO_DIR, label_dir=DEFAULT_LABEL_DIR):
    """
    Finds and pairs audio files with their corresponding label files.
    Matches are based on the base filename (excluding extensions).
    """
    # Supported audio extensions
    audio_extensions = ["*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
        
    pairs = []
    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Suffixes to check and strip for matching label files
        # (e.g., GuitarSet audio files end with '_mic' but label files do not)
        possible_names = [base_name]
        for suffix in ["_mic", "_mix"]:
            if base_name.endswith(suffix):
                possible_names.append(base_name[:-len(suffix)])
        
        matched_label = None
        for name in possible_names:
            label_paths = [
                os.path.join(label_dir, f"{name}.lab"),
                os.path.join(label_dir, f"{name}.txt")
            ]
            for lp in label_paths:
                if os.path.exists(lp):
                    matched_label = lp
                    break
            if matched_label:
                break
                
        if matched_label:
            pairs.append((audio_path, matched_label))
            
    return pairs

def train_pipeline(audio_dir=DEFAULT_AUDIO_DIR, label_dir=DEFAULT_LABEL_DIR, epochs=30, batch_size=32):
    """
    Runs the full end-to-end training pipeline.
    """
    print("=== Step 1: Finding Dataset Files ===")
    pairs = find_dataset_files(audio_dir, label_dir)
    print(f"Found {len(pairs)} matched audio/label pairs.")
    
    if not pairs:
        print("WARNING: No matched audio and label files found!")
        print(f"Please place your audio files in '{audio_dir}' and label files in '{label_dir}'.")
        print("Make sure they share the same base name (e.g. 'track1.wav' and 'track1.lab').")
        return None
        
    print("\n=== Step 2: Extracting CQT Chroma Features & Aligning Labels ===")
    X_list = []
    y_list = []
    
    for audio_path, label_path in pairs:
        print(f"Preprocessing: {os.path.basename(audio_path)} ...")
        features, labels = preprocess_track(audio_path, label_path, window_size=WINDOW_SIZE)
        X_list.append(features)
        y_list.append(labels)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Preprocessing complete. Total frames extracted: {X.shape[0]}")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # 3. Train-test split (80% train, 20% validation)
    print("\n=== Step 3: Splitting Dataset ===")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]} frames | Validation size: {X_val.shape[0]} frames")
    
    # 4. Build and compile CNN model
    print("\n=== Step 4: Building CNN Model ===")
    model = build_chord_cnn(num_classes=25, window_size=WINDOW_SIZE)
    model.summary()
    
    # 5. Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True,
        verbose=1
    )
    
    # 6. Fit model
    print("\n=== Step 5: Training CNN Model ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # 7. Save history metrics to a JSON file
    history_dict = history.history
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training metrics saved to '{HISTORY_SAVE_PATH}'.")
    
    # 8. Optional: Plot training history if matplotlib is available
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 5))
        
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = "training_curves.png"
        plt.savefig(plot_path)
        print(f"Saved training curve plot to '{plot_path}'.")
    else:
        print("Note: Install matplotlib (`pip install matplotlib`) to automatically generate training curve plots.")
        
    return model

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs(DEFAULT_AUDIO_DIR, exist_ok=True)
    os.makedirs(DEFAULT_LABEL_DIR, exist_ok=True)
    
    # Run the training pipeline on the full GuitarSet dataset
    train_pipeline(epochs=20, batch_size=64)
