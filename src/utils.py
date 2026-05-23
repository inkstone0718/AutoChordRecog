# -*- coding: utf-8 -*-
"""
utils.py - Chord vocabulary and simplification utilities.
Maps raw chord annotations to a standard 25-class vocabulary:
12 Major chords, 12 Minor chords, and 1 No-chord (N).
"""

# Define the standard 25-chord vocabulary
MAJORS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MINORS = [ch + 'm' for ch in MAJORS]
CHORDS = MAJORS + MINORS + ['N']

# Create mapping dictionaries
CHORD_TO_IDX = {chord: idx for idx, chord in enumerate(CHORDS)}
IDX_TO_CHORD = {idx: chord for idx, chord in enumerate(CHORDS)}

# Mapping from flat notes to sharp notes
FLAT_TO_SHARP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'Cb': 'B',  'Fb': 'E',  'E#': 'F',  'B#': 'C'
}

def simplify_chord(chord_str):
    """
    Simplifies a raw chord label (e.g. 'C:maj7', 'D#min/5', 'Db', 'N') 
    into one of the 25 standard classes in the vocabulary.
    """
    if not chord_str:
        return 'N'
    
    # Strip spaces
    chord_str = chord_str.strip()
    
    # Check for No-chord or silence
    if chord_str == 'N' or chord_str == '' or chord_str.upper() == 'X':
        return 'N'
        
    # Standardize separator if any (e.g., Harte notation C:maj7 -> C)
    parts = chord_str.split(':')
    root_part = parts[0]
    suffix_part = parts[1] if len(parts) > 1 else ''
    
    # If root_part is empty, default to 'N'
    if not root_part:
        return 'N'
        
    # Extract root (first character must be A-G)
    base_note = root_part[0].upper()
    if base_note not in 'ABCDEFG':
        return 'N'
        
    # Check for accidental (sharp or flat)
    accidental = ''
    if len(root_part) > 1 and root_part[1] in ['#', 'b']:
        accidental = root_part[1]
        
    root = base_note + accidental
    
    # Map flats to sharps or other equivalents
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
        
    # Determine the suffix part
    # Combine the rest of the root_part (if any, after accidental) with the suffix_part
    remaining_suffix = root_part[len(base_note + accidental):] + suffix_part
    suffix = remaining_suffix.lower()
    
    # Determine if it's minor
    # dim (diminished) and hdim (half-diminished) are mapped to minor for simplicity
    is_minor = False
    if 'min' in suffix or 'dim' in suffix or 'hdim' in suffix:
        is_minor = True
    elif suffix.startswith('m') and not suffix.startswith('maj'):
        is_minor = True
        
    if is_minor:
        return root + 'm'
    else:
        return root

if __name__ == '__main__':
    # Test cases
    test_chords = [
        'C', 'C:maj', 'C:maj7', 'C#', 'Db', 'D:min7', 'D#m', 'Eb:min', 
        'F:dim', 'G:hdim7', 'A:maj/5', 'N', 'X', 'Bb', 'B:7'
    ]
    print("Testing chord simplification:")
    for tc in test_chords:
        simplified = simplify_chord(tc)
        idx = CHORD_TO_IDX.get(simplified, -1)
        print(f"Raw: {tc:<10} -> Simplified: {simplified:<6} -> Index: {idx}")
