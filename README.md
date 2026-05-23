# Automatic Chord Recognition (ACR) Project

This repository contains a modular machine learning project designed to recognize musical chords from audio files. It extracts Constant-Q Transform (CQT) chroma features from audio files, aligns them with annotation labels, trains a 2D Convolutional Neural Network (CNN) in Keras 3/TensorFlow, and produces a transcribed chord timeline.

---

## 📂 Project Directory Structure

```text
├── .venv/                   # Python virtual environment (modules installed here)
├── data/
│   ├── audio/               # Place your audio files (.wav, .mp3, .ogg, etc.) here
│   └── labels/              # Place corresponding label files (.lab, .txt) here
├── src/
│   ├── __init__.py
│   ├── utils.py             # Chord vocabulary (25 classes) and simplification mapping
│   ├── preprocess.py        # Extracts chroma features and aligns timestamps to labels
│   ├── model.py             # 2D CNN model definition (built for Keras 3 / TensorFlow 2.16+)
│   ├── train.py             # Pipeline to preprocess, split, train, and save the model
│   └── predict.py           # Runs prediction on new audio and outputs a chord timeline
├── scripts/
│   ├── download_guitarset.py# Automation to download and prepare GuitarSet
│   └── record_and_predict.py# Interactive microphone recorder and chord predictor
├── ChordRecog.py            # Legacy single-file test script
└── README.md                # Project documentation and guide (this file)
```

---

## 🛠️ Getting Started & Setup

Since you are running inside a Python 3.12 virtual environment, you can run all scripts using the virtual environment interpreter.

### 1. Activating the Virtual Environment
To work inside your terminal with the virtual environment activated:
```bash
source .venv/bin/activate
```
*(Once activated, you can just use `python` and `pip` directly. To exit, type `deactivate`.)*

### 2. Dataset Preparation
To train a model on your own recordings:
1. Place your audio tracks (e.g. `song1.wav`, `guitar_rif.mp3`) in **`data/audio/`**.
2. Place the matching chord annotation files (e.g. `song1.lab`, `guitar_rif.lab` or `.txt`) in **`data/labels/`**.
3. **Important**: The audio file and its label file must share the same base name (e.g., `guitar.wav` and `guitar.lab`).

#### Annotation Label Format
The label files should be space- or tab-delimited text containing the start time, end time, and chord:
```text
0.00 2.50 C
2.50 5.10 G
5.10 7.80 Am
7.80 10.20 F
```
*(The system automatically maps complex chord names like `Cmaj7` or `Dsus4` to standard Major/Minor root equivalents: `C` or `D`).*

---

## 🚀 Running the Pipeline

### Step 1: Train the Model
To preprocess your files and train the neural network:
```bash
.venv/bin/python src/train.py
```
This script will:
* Pair your audio and label files.
* Extract windowed CQT Chroma features.
* Split the dataset into 80% training and 20% validation.
* Train the model for 30 epochs (with Early Stopping if validation loss stops improving).
* Save the best weights to `best_chord_model.keras`.
* Save training metrics to `training_history.json`.

### Step 2: Predict Chords on New Audio
To recognize chords on a new audio file using your trained model:
```bash
.venv/bin/python src/predict.py <path_to_audio_file>
```
For example, to test with the dummy trumpet file:
```bash
.venv/bin/python src/predict.py data/audio/trumpet.wav
```
This will print a clean, merged chord transcription timeline:
```text
=== Predicted Chord Timeline ===
Start (s)  | End (s)    | Chord 
----------------------------------
0.00       | 1.39       | C     
1.39       | 2.79       | G     
2.79       | 3.72       | Am    
3.72       | 5.39       | F     
```

### Step 3: Record and Predict in Real-Time (macOS)
To record yourself playing chords on the guitar and immediately run transcription:
```bash
.venv/bin/python scripts/record_and_predict.py
```
This script will:
* Count down `3, 2, 1...` so you can prepare your hands.
* Start recording from your default microphone using `ffmpeg` (stops when you press **[ENTER]**).
* Save the audio to `my_recording.wav`.
* Instantly load `best_chord_model.keras` and print out your transcribed timeline!

---

## 🎓 Beginner's Guide & Next Steps in ML

As you develop and expand this project, here is how you can improve it step-by-step:

### 1. Data Augmentation
ML models need plenty of data to generalize. For audio, you can augment your training set by:
* **Pitch Shifting**: Shift the pitch of your audio up/down by semitones (e.g. using `librosa.effects.pitch_shift`) and adjust the labels accordingly (e.g., if you shift `C` up 2 semitones, it becomes `D`).
* **Time Stretching**: Speed up or slow down the audio slightly without changing the pitch.

### 2. Handling Temporal Context (CRNN)
Chords don't change randomly; they follow musical progressions. 
* Currently, the model looks at an isolated window of 11 frames (~1 second) to predict a chord.
* **Next Level**: You can use a **Convolutional Recurrent Neural Network (CRNN)**. It first processes frames with a CNN, and then feeds those features into an **LSTM** or **GRU** layer to learn long-term sequences and dependencies in the song.

### 3. Output Smoothing (HMM / Viterbi)
Sometimes a model might predict `C - C - G - C - C` in consecutive frames due to noise. You can use a **Hidden Markov Model (HMM)** to smooth the output sequence. The HMM uses chord transition probabilities (e.g., how likely is a transition from `G` to `C`) to calculate the most musically coherent sequence.
