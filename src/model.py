# -*- coding: utf-8 -*-
"""
model.py - CNN Architecture for Automatic Chord Recognition.
Defines a 2D Convolutional Neural Network (CNN) specifically tailored
to process 12x11 CQT chroma feature windows.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_chord_cnn(num_classes=25, window_size=11):
    """
    Constructs a 2D CNN model for processing CQT chroma feature frames.
    
    Parameters:
        num_classes (int): Number of chord classes to predict (default: 25).
        window_size (int): Width of the temporal context window (default: 11).
    """
    model = models.Sequential([
        # Keras 3 Input definition
        layers.Input(shape=(12, window_size, 1)),
        
        # Block 1: Conv -> BN -> Pool
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Conv -> BN -> Dropout
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fully Connected Block
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Classification Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    # Print the model summary to verify setup
    print("Building model...")
    model = build_chord_cnn(num_classes=25, window_size=11)
    model.summary()
    print("Model compilation verification successful!")
