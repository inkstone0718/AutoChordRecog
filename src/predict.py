import os
import argparse
import numpy as np
import tensorflow as tf
import librosa

try:
    from src.preprocess import preprocess_track, HOP_LENGTH, SR, WINDOW_SIZE
    from src.utils import IDX_TO_CHORD, CHORD_TO_IDX
except ImportError:
    from preprocess import preprocess_track, HOP_LENGTH, SR, WINDOW_SIZE
    from utils import IDX_TO_CHORD, CHORD_TO_IDX

TRIAD_MAP = {
    'C': ['c/4', 'e/4', 'g/4'],
    'C#': ['c#/4', 'f/4', 'g#/4'],
    'D': ['d/4', 'f#/4', 'a/4'],
    'D#': ['d#/4', 'g/4', 'a#/4'],
    'E': ['e/4', 'g#/4', 'b/4'],
    'F': ['f/4', 'a/4', 'c/5'],
    'F#': ['f#/4', 'a#/4', 'c#/5'],
    'G': ['g/4', 'b/4', 'd/5'],
    'G#': ['g#/4', 'c/5', 'd#/5'],
    'A': ['a/4', 'c#/5', 'e/5'],
    'A#': ['a#/4', 'd/5', 'f/5'],
    'B': ['b/4', 'd#/5', 'f#/5'],
    'Cm': ['c/4', 'eb/4', 'g/4'],
    'C#m': ['c#/4', 'e/4', 'g#/4'],
    'Dm': ['d/4', 'f/4', 'a/4'],
    'D#m': ['d#/4', 'f#/4', 'a#/4'],
    'Em': ['e/4', 'g/4', 'b/4'],
    'Fm': ['f/4', 'ab/4', 'c/5'],
    'F#m': ['f#/4', 'a/4', 'c#/5'],
    'Gm': ['g/4', 'bb/4', 'd/5'],
    'G#m': ['g#/4', 'b/4', 'd#/5'],
    'Am': ['a/4', 'c/5', 'e/5'],
    'A#m': ['a#/4', 'c#/5', 'f/5'],
    'Bm': ['b/4', 'd/5', 'f#/5'],
    'N': ['b/4']
}

TAB_MAP = {
    'C': [{'str': 5, 'fret': 3}, {'str': 4, 'fret': 2}, {'str': 3, 'fret': 0}, {'str': 2, 'fret': 1}, {'str': 1, 'fret': 0}],
    'C#': [{'str': 5, 'fret': 4}, {'str': 4, 'fret': 6}, {'str': 3, 'fret': 6}, {'str': 2, 'fret': 6}, {'str': 1, 'fret': 4}],
    'D': [{'str': 4, 'fret': 0}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 3}, {'str': 1, 'fret': 2}],
    'D#': [{'str': 5, 'fret': 6}, {'str': 4, 'fret': 8}, {'str': 3, 'fret': 8}, {'str': 2, 'fret': 8}, {'str': 1, 'fret': 6}],
    'E': [{'str': 6, 'fret': 0}, {'str': 5, 'fret': 2}, {'str': 4, 'fret': 2}, {'str': 3, 'fret': 1}, {'str': 2, 'fret': 0}, {'str': 1, 'fret': 0}],
    'F': [{'str': 6, 'fret': 1}, {'str': 5, 'fret': 3}, {'str': 4, 'fret': 3}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 1}, {'str': 1, 'fret': 1}],
    'F#': [{'str': 6, 'fret': 2}, {'str': 5, 'fret': 4}, {'str': 4, 'fret': 4}, {'str': 3, 'fret': 3}, {'str': 2, 'fret': 2}, {'str': 1, 'fret': 2}],
    'G': [{'str': 6, 'fret': 3}, {'str': 5, 'fret': 2}, {'str': 4, 'fret': 0}, {'str': 3, 'fret': 0}, {'str': 2, 'fret': 0}, {'str': 1, 'fret': 3}],
    'G#': [{'str': 6, 'fret': 4}, {'str': 5, 'fret': 6}, {'str': 4, 'fret': 6}, {'str': 3, 'fret': 5}, {'str': 2, 'fret': 4}, {'str': 1, 'fret': 4}],
    'A': [{'str': 5, 'fret': 0}, {'str': 4, 'fret': 2}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 2}, {'str': 1, 'fret': 0}],
    'A#': [{'str': 5, 'fret': 1}, {'str': 4, 'fret': 3}, {'str': 3, 'fret': 3}, {'str': 2, 'fret': 3}, {'str': 1, 'fret': 1}],
    'B': [{'str': 5, 'fret': 2}, {'str': 4, 'fret': 4}, {'str': 3, 'fret': 4}, {'str': 2, 'fret': 4}, {'str': 1, 'fret': 2}],
    'Cm': [{'str': 5, 'fret': 3}, {'str': 4, 'fret': 5}, {'str': 3, 'fret': 5}, {'str': 2, 'fret': 4}, {'str': 1, 'fret': 3}],
    'C#m': [{'str': 5, 'fret': 4}, {'str': 4, 'fret': 6}, {'str': 3, 'fret': 6}, {'str': 2, 'fret': 5}, {'str': 1, 'fret': 4}],
    'Dm': [{'str': 4, 'fret': 0}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 3}, {'str': 1, 'fret': 1}],
    'D#m': [{'str': 5, 'fret': 6}, {'str': 4, 'fret': 8}, {'str': 3, 'fret': 8}, {'str': 2, 'fret': 7}, {'str': 1, 'fret': 6}],
    'Em': [{'str': 6, 'fret': 0}, {'str': 5, 'fret': 2}, {'str': 4, 'fret': 2}, {'str': 3, 'fret': 0}, {'str': 2, 'fret': 0}, {'str': 1, 'fret': 0}],
    'Fm': [{'str': 6, 'fret': 1}, {'str': 5, 'fret': 3}, {'str': 4, 'fret': 3}, {'str': 3, 'fret': 1}, {'str': 2, 'fret': 1}, {'str': 1, 'fret': 1}],
    'F#m': [{'str': 6, 'fret': 2}, {'str': 5, 'fret': 4}, {'str': 4, 'fret': 4}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 2}, {'str': 1, 'fret': 2}],
    'Gm': [{'str': 6, 'fret': 3}, {'str': 5, 'fret': 5}, {'str': 4, 'fret': 5}, {'str': 3, 'fret': 3}, {'str': 2, 'fret': 3}, {'str': 1, 'fret': 3}],
    'G#m': [{'str': 6, 'fret': 4}, {'str': 5, 'fret': 6}, {'str': 4, 'fret': 6}, {'str': 3, 'fret': 4}, {'str': 2, 'fret': 4}, {'str': 1, 'fret': 4}],
    'Am': [{'str': 5, 'fret': 0}, {'str': 4, 'fret': 2}, {'str': 3, 'fret': 2}, {'str': 2, 'fret': 1}, {'str': 1, 'fret': 0}],
    'A#m': [{'str': 5, 'fret': 1}, {'str': 4, 'fret': 3}, {'str': 3, 'fret': 3}, {'str': 2, 'fret': 2}, {'str': 1, 'fret': 1}],
    'Bm': [{'str': 5, 'fret': 2}, {'str': 4, 'fret': 4}, {'str': 3, 'fret': 4}, {'str': 2, 'fret': 3}, {'str': 1, 'fret': 2}],
    'N': [{'str': 3, 'fret': 0}]
}

def quantize_timeline_to_measures(timeline):
    """
    Quantizes the chord timeline to a 120 BPM grid (0.5s per beat), groups them
    into perfect 4/4 measures, and returns a list of measures with VexFlow-compatible note definitions.
    """
    BPM = 120
    beat_dur = 60.0 / BPM
    
    total_duration = timeline[-1][1] if timeline else 0
    num_beats = int(round(total_duration / beat_dur))
    if num_beats == 0:
        num_beats = 4
        
    beat_chords = []
    for i in range(num_beats):
        t_mid = (i + 0.5) * beat_dur
        chord_found = 'N'
        for start, end, chord in timeline:
            if start <= t_mid <= end:
                chord_found = chord
                break
        else:
            if timeline:
                chord_found = timeline[-1][2]
        beat_chords.append(chord_found)
        
    measures = []
    for i in range(0, len(beat_chords), 4):
        measure_chords = beat_chords[i:i+4]
        if len(measure_chords) < 4:
            measure_chords += ['N'] * (4 - len(measure_chords))
        measures.append(measure_chords)
        
    js_measures = []
    for measure_chords in measures:
        js_measure = []
        segments = []
        current_chord = measure_chords[0]
        current_dur = 1
        for chord in measure_chords[1:]:
            if chord == current_chord:
                current_dur += 1
            else:
                segments.append((current_chord, current_dur))
                current_chord = chord
                current_dur = 1
        segments.append((current_chord, current_dur))
        
        for chord, dur in segments:
            if dur == 3:
                notes_to_add = [(chord, 2), (chord, 1)]
            else:
                notes_to_add = [(chord, dur)]
                
            for ch, d in notes_to_add:
                dur_map = {4: "w", 2: "h", 1: "q"}
                vex_dur = dur_map.get(d, "q")
                is_rest = (ch == 'N')
                keys = TRIAD_MAP.get(ch, ['b/4'])
                positions = TAB_MAP.get(ch, [{'str': 3, 'fret': 0}])
                
                js_measure.append({
                    "chord": ch,
                    "keys": keys,
                    "positions": positions,
                    "duration": vex_dur,
                    "isRest": is_rest
                })
        js_measures.append(js_measure)
    return js_measures


def generate_html_sheet_music(timeline, output_path="sheet_music.html"):
    """
    Writes a premium, interactive HTML page that renders the sheet music
    and plays the chord triads using Web Audio synthesis. Includes a dynamic recording
    button that triggers recording and re-transcription.
    """
    BPM = 120
    js_measures = quantize_timeline_to_measures(timeline)
    import json
    measures_json = json.dumps(js_measures)
    
    # Load template from file
    template_path = os.path.join(os.path.dirname(__file__), "sheet_music_template.html")
    if not os.path.exists(template_path):
        template_path = "src/sheet_music_template.html"
    
    with open(template_path, "r", encoding="utf-8") as tf:
        template_content = tf.read()
        
    html_content = template_content.replace("{BPM}", str(BPM)).replace("{measures_json}", measures_json)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n👉 Visualized sheet music saved to '{output_path}'. Open this file in your browser to view it!")


def predict_chords(audio_path, model_path="best_chord_model.keras", smoothing_window=7, min_duration=0.4, silence_threshold=0.015):
    """
    Transcribes the chords in an audio file, applies temporal smoothing, silence gating,
    and duration filtering, prints a timeline, and generates visual sheet music.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train a model first using: python src/train.py")
        return None
        
    print(f"=== Loading Model: {model_path} ===")
    model = tf.keras.models.load_model(model_path)
    
    print(f"=== Processing Audio: {os.path.basename(audio_path)} ===")
    features, frame_times = preprocess_track(audio_path, label_path=None, window_size=WINDOW_SIZE)
    print(f"Extracted {features.shape[0]} feature frames.")
    
    # Run prediction
    print("Running inference...")
    predictions = model.predict(features, verbose=0)
    
    # 1. Silence Gating using RMS Energy
    y, sr = librosa.load(audio_path, sr=SR)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]
    
    # Align RMS energy length with frame times
    if len(rms) < len(frame_times):
        rms = np.pad(rms, (0, len(frame_times) - len(rms)), mode='edge')
    else:
        rms = rms[:len(frame_times)]
        
    # 2. Probability Moving Average Temporal Smoothing
    if smoothing_window > 1:
        smoothed_preds = np.zeros_like(predictions)
        half_w = smoothing_window // 2
        for i in range(len(predictions)):
            start = max(0, i - half_w)
            end = min(len(predictions), i + half_w + 1)
            smoothed_preds[i] = np.mean(predictions[start:end], axis=0)
        predictions = smoothed_preds
        
    pred_indices = np.argmax(predictions, axis=1)
    
    # Apply silence gate
    for i, energy in enumerate(rms):
        if energy < silence_threshold:
            pred_indices[i] = CHORD_TO_IDX['N']
            
    # 3. Merge identical frames into raw segments
    raw_segments = []
    current_chord = None
    frame_duration = HOP_LENGTH / SR
    
    for i, idx in enumerate(pred_indices):
        chord_name = IDX_TO_CHORD[idx]
        frame_time = frame_times[i]
        
        if current_chord is None:
            current_chord = chord_name
            start_time = frame_time
        elif chord_name != current_chord:
            end_time = frame_time
            raw_segments.append([start_time, end_time, current_chord])
            current_chord = chord_name
            start_time = frame_time
            
    if current_chord is not None:
        end_time = frame_times[-1] + frame_duration
        raw_segments.append([start_time, end_time, current_chord])
        
    # 4. Filter segments below minimum duration
    filtered_segments = []
    for seg in raw_segments:
        duration = seg[1] - seg[0]
        if duration >= min_duration or seg[2] == 'N':  # Keep silence 'N' segments
            filtered_segments.append(seg)
        else:
            # Merge with the previous segment if available
            if filtered_segments:
                filtered_segments[-1][1] = seg[1]
            else:
                filtered_segments.append(seg)
                
    # Merge consecutive identical chords after duration filtering
    timeline = []
    for seg in filtered_segments:
        if not timeline:
            timeline.append((seg[0], seg[1], seg[2]))
        else:
            if seg[2] == timeline[-1][2]:
                # Extend the previous segment
                timeline[-1] = (timeline[-1][0], seg[1], timeline[-1][2])
            else:
                timeline.append((seg[0], seg[1], seg[2]))
                
    print("\n=== Predicted Chord Timeline ===")
    print(f"{'Start (s)':<10} | {'End (s)':<10} | {'Chord':<6}")
    print("-" * 34)
    for start, end, chord in timeline:
        print(f"{start:<10.2f} | {end:<10.2f} | {chord:<6}")
        
    # Print the one-dimensional visual timeline starting from the first non-'N' chord
    first_non_n_idx = -1
    for idx, (start, end, chord) in enumerate(timeline):
        if chord != 'N':
            first_non_n_idx = idx
            break
            
    if first_non_n_idx != -1:
        scale_start, scale_end, scale_chord = timeline[first_non_n_idx]
        scale_dur = scale_end - scale_start
        char_scale = 12  # Number of characters representing 1x of the scale duration
        
        visual_parts = []
        for i in range(first_non_n_idx, len(timeline)):
            start, end, chord = timeline[i]
            dur = end - start
            
            if chord == 'N':
                # Silence represented with shading characters
                width = max(1, int(round((dur / scale_dur) * char_scale)))
                visual_parts.append("░" * width)
            else:
                # Chord block represented with brackets
                min_width = len(chord) + 2
                width = max(min_width, int(round((dur / scale_dur) * char_scale)))
                
                label = f" {chord} "
                padding = width - len(label) - 2  # -2 for brackets
                if padding >= 0:
                    pad_left = padding // 2
                    pad_right = padding - pad_left
                    block = f"[{' ' * pad_left}{label}{' ' * pad_right}]"
                else:
                    # Fallback to compact representation if padding is negative
                    block = f"[{chord}]"
                visual_parts.append(block)
                
        visual_line = "".join(visual_parts)
        
        print("\n=== One-Dimensional Visual Timeline ===")
        print(f"Scale: 1 Unit [ {'=' * (char_scale - 4)} ] = {scale_dur:.2f}s (duration of first chord '{scale_chord}')")
        print(visual_line)
        
    # Generate HTML sheet music
    try:
        generate_html_sheet_music(timeline)
    except Exception as e:
        print(f"Warning: Failed to generate HTML sheet music due to: {e}")
        
    return timeline

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe chords from an audio file.")
    parser.add_argument("audio", type=str, nargs="?", default=None, help="Path to the input audio file.")
    parser.add_argument("--model", type=str, default="best_chord_model.keras", help="Path to the trained keras model.")
    parser.add_argument("--smoothing", type=int, default=7, help="Moving average smoothing window size (odd integer).")
    parser.add_argument("--min-duration", type=float, default=0.4, help="Minimum duration of a chord segment in seconds.")
    parser.add_argument("--silence-threshold", type=float, default=0.015, help="RMS energy threshold below which chord is classified as silence 'N'.")
    
    args = parser.parse_args()
    
    if args.audio is None:
        print("Usage: python src/predict.py <path_to_audio_file> [options]")
        print("No audio file provided. Running test prediction using librosa's built-in trumpet sample...")
        test_audio_path = librosa.ex('trumpet')
        predict_chords(test_audio_path, model_path=args.model, smoothing_window=args.smoothing, min_duration=args.min_duration, silence_threshold=args.silence_threshold)
    else:
        predict_chords(args.audio, model_path=args.model, smoothing_window=args.smoothing, min_duration=args.min_duration, silence_threshold=args.silence_threshold)

