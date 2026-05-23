# Guide: Realistic Guitar Sound Generation and Synthesis

This document outlines the core principles, parameters, and algorithms required to synthesize or generate realistic-sounding acoustic guitar chords and notes. Use these guidelines to design sound generators for chord ear-training web apps or for generating synthetic data for auto chord recognition ML projects.

---

## 1. Core Principle: Staggered Strumming (Strum Emulation)

### The Problem
When a human plays a chord on a guitar, they sweep their pick or fingers across the strings sequentially. If you play all notes in a chord at the exact same millisecond (a "block chord"), it sounds mechanical, artificial, and like a keyboard synthesizer.

### The Solution
Implement a **staggered start time** for each note in the chord.
- **Strum Delay**: Introduce a micro-delay between the trigger times of consecutive strings.
- **Optimal Delay Value**: A delay of **$30\text{ ms}$ to $50\text{ ms}$** (0.03 to 0.05 seconds) per string simulates a standard, natural strum. We found **$40\text{ ms}$ (0.04s)** to yield the most natural acoustic resonance.
- **Strum Directions**:
  - **Downstrum** (default / standard): Play strings from lowest pitch to highest pitch (ascending array index).
  - **Upstrum**: Play strings from highest pitch to lowest pitch (descending array index).
  - **Strum Speed**: Varying the delay (e.g., $10\text{ ms}$ for fast, aggressive strums; $80\text{ ms}$ for slow arpeggiated rolls) creates different playing dynamics.

---

## 2. Guitar-Specific Voicings (Chord Tabs)

### The Problem
Standard keyboard chords use close-position triads (e.g., C Major: $C4 \rightarrow E4 \rightarrow G4$). A real guitar has 6 strings (tuned $E2, A2, D3, G3, B3, E4$), and chords are played using specific fingering patterns across frets. Triads sound thin and incorrect for a guitar.

### The Solution
Define chords using authentic guitar voicings that match physical tab patterns. Below are example mappings implemented in our system:

| Chord | Tab Fingering ($E\ A\ D\ G\ B\ E$) | MIDI Notes / Pitch Names |
|---|---|---|
| **C Major** | `X32010` | `['C3', 'E3', 'G3', 'C4', 'E4']` |
| **G Major** | `320003` | `['G2', 'B2', 'D3', 'G3', 'B3', 'G4']` |
| **E Minor** | `022000` | `['E2', 'B2', 'E3', 'G3', 'B3', 'E4']` |
| **A Minor** | `X02210` | `['A2', 'E3', 'A3', 'C4', 'E4']` |
| **D Major** | `XX0232` | `['D3', 'A3', 'D4', 'F#4']` |
| **F Major (Barre)**| `133211` | `['F2', 'C3', 'F3', 'A3', 'C4', 'F4']` |
| **C maj7** | `X32000` | `['C3', 'E3', 'G3', 'B3', 'E4']` |

> [!TIP]
> Always filter out strings marked with `X` (muted) when generating the note list. For standard chords, ensure the root note is the lowest pitch played.

---

## 3. High-Quality Sample-Based Synthesis (Soundfonts / Samplers)

For a realistic acoustic tone, simple oscillator waveforms (sine, triangle, square) are inadequate because they lack the harmonic richness and complex decay of vibrating metal/nylon strings.

### Web / JavaScript Stack
Use **Soundfonts** (specifically the General MIDI steel acoustic guitar sample set) to load realistic waveforms.
- **Library**: `soundfont-player` (coupled with Web Audio API's `AudioContext`).
- **Instrument**: `'acoustic_guitar_steel'` or `'acoustic_guitar_nylon'`.

```typescript
// Example: Staggered chord playback in JS/TS using soundfont-player
import Soundfont from 'soundfont-player';

let instrument: any = null;

export const initAudio = async () => {
  if (!instrument) {
    const ac = new (window.AudioContext || (window as any).webkitAudioContext)();
    instrument = await Soundfont.instrument(ac, 'acoustic_guitar_steel');
  }
};

export const playChord = (notes: string[]) => {
  if (!instrument) return;
  const now = instrument.context.currentTime;
  
  // Stagger each note trigger by 40ms to simulate a strum
  notes.forEach((note, i) => {
    instrument.play(note, now + i * 0.04, { duration: 2.5, gain: 2.0 });
  });
};
```

### Python Stack (For ML Data Augmentation & Synthesis)
When generating synthetic audio files for training ML models:
1. Use **MIDI files** with specific note layouts, staggered start times, and velocities.
2. Render MIDI to audio (`.wav`) using **FluidSynth** and a high-quality Soundfont file (`.sf2` format).
- **Libraries**: `pretty_midi` (for creating MIDIs) and `pyfluidsynth` or the CLI `fluidsynth` (for rendering).

```python
# Example: Python synthesis of a strummed guitar chord
import pretty_midi
import subprocess

def create_strummed_midi(chord_notes, output_midi_path, strum_delay=0.04):
    pm = pretty_midi.PrettyMIDI()
    # Steel guitar program number is 25 (Acoustic Guitar - steel)
    instrument = pretty_midi.Instrument(program=25)
    
    start_time = 0.0
    duration = 2.5
    
    # Stagger note start times
    for i, pitch_name in enumerate(chord_notes):
        pitch = pretty_midi.key_name_to_number(pitch_name)
        note_start = start_time + (i * strum_delay)
        note_end = start_time + duration
        
        # Add micro-variations to velocity (default 100) to simulate human dynamics
        velocity = 95 + (i * 2) # slightly accentuating higher strings or adding randomness
        
        note = pretty_midi.Note(
            velocity=velocity, 
            pitch=pitch, 
            start=note_start, 
            end=note_end
        )
        instrument.notes.append(note)
        
    pm.instruments.append(instrument)
    pm.write(output_midi_path)

def render_midi_to_wav(midi_path, soundfont_path, output_wav_path):
    # Call fluidsynth command line to render MIDI to CD-quality audio
    cmd = [
        "fluidsynth", "-F", output_wav_path, "-T", "wav", soundfont_path, midi_path
    ]
    subprocess.run(cmd, check=True)
```

---

## 4. Synthesis via Physical Modeling (Karplus-Strong Algorithm)

If samples/Soundfonts cannot be used, implement the **Karplus-Strong algorithm**. This is a digital waveguide physical modeling synthesis method that models a plucked string using a short feedback loop.

### Algorithm Flow
1. **Excitation**: Fill a delay line of length $N$ with a brief burst of white noise (simulating the initial pluck).
   - The delay line length $N$ determines the fundamental frequency: $N = \frac{f_s}{f}$, where $f_s$ is the sample rate (e.g., $44100\text{ Hz}$) and $f$ is the target note frequency.
2. **Feedback Loop**: Output the first sample of the delay line, pass it through a low-pass filter (averaging adjacent samples to model string damping), and feed it back into the end of the delay line.
   - Equation: $y[n] = x[n] + \alpha \cdot \frac{y[n-N] + y[n-N-1]}{2}$, where $\alpha \approx 0.996$ represents string decay.
3. **Strum Simulation**: Create multiple independent physical modeling string instances and excite them with the $40\text{ ms}$ delay offset.
