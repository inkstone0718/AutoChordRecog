# -*- coding: utf-8 -*-
"""
server.py - Lightweight local web server for the Guitar Chord Sheet Music application.
Serves sheet_music.html and exposes endpoints to control background audio recording and ML prediction.
"""

import os
import sys
import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add the project root directory to the python path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_chords, quantize_timeline_to_measures

# Global state for recording process
recording_process = None
recording_file = "my_recording.wav"

class ChordRecServer(BaseHTTPRequestHandler):
    def do_GET(self):
        # Resolve path
        path = self.path.split('?')[0]
        if path in ("/", "/sheet_music.html"):
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sheet_music.html"))
            
            # If sheet_music.html does not exist, run a default prediction to create it
            if not os.path.exists(file_path):
                print("sheet_music.html not found. Generating default sheet music from sample...")
                try:
                    # Import librosa's built-in trumpet sample or run prediction on an empty run
                    import librosa
                    test_audio_path = librosa.ex('trumpet')
                    predict_chords(test_audio_path, model_path="best_chord_model.keras")
                except Exception as e:
                    print(f"Could not generate default sheet_music.html: {e}")
                    self.send_error(500, f"Could not generate sheet_music.html: {e}")
                    return
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # Disable browser caching
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        elif path in ("/favicon.ico", "/.well-known/appspecific/com.chrome.devtools.json"):
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        global recording_process
        
        path = self.path.split('?')[0]
        if path == "/start_record":
            # If a recording is already running, clean it up
            if recording_process is not None:
                try:
                    recording_process.terminate()
                    recording_process.wait()
                except Exception:
                    pass
                recording_process = None
                
            print("\n🔴 RECORDING STARTED via browser button request...")
            
            # Start ffmpeg in background to capture macOS default microphone
            cmd = [
                "ffmpeg",
                "-f", "avfoundation",
                "-i", ":default",
                "-y",
                recording_file
            ]
            
            try:
                recording_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Recording started"}).encode('utf-8'))
            except Exception as e:
                print(f"Failed to start ffmpeg: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
                
        elif path == "/stop_record":
            if recording_process is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "No active recording session"}).encode('utf-8'))
                return
                
            print("Stopping recording...")
            try:
                # Send 'q' to ffmpeg to stop gracefully
                recording_process.communicate(input=b'q', timeout=2)
            except Exception as e:
                print(f"Error stopping ffmpeg gracefully: {e}. Terminating process...")
                try:
                    recording_process.terminate()
                    recording_process.wait()
                except Exception:
                    pass
            finally:
                recording_process = None
                
            print(f"Recording saved to '{recording_file}'")
            
            if not os.path.exists(recording_file):
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Audio file was not recorded successfully."}).encode('utf-8'))
                return
                
            print("Running inference on recorded audio...")
            try:
                # Run the prediction pipeline
                timeline = predict_chords(
                    recording_file,
                    model_path="best_chord_model.keras",
                    smoothing_window=7,
                    min_duration=0.4,
                    silence_threshold=0.015
                )
                
                # Quantize the timeline to measures
                measures = quantize_timeline_to_measures(timeline)
                
                # Respond with the quantized measures JSON
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(json.dumps(measures).encode('utf-8'))
                print("Prediction complete and sent back successfully!")
            except Exception as e:
                print(f"Prediction failed: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Transcription failed: {str(e)}"}).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChordRecServer)
    print(f"🚀 Chord Recognition Server running at http://localhost:{port}/")
    print("👉 Open your browser to http://localhost:8000/ to view and record sheet music.")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run()
