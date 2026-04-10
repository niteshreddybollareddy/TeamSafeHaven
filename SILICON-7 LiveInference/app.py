"""
SILICON-7 Live Inference Web Application
Flask backend for real-time audio classification
"""

import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime
import shutil

from inference import AudioInference

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Storage directories
STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'storage')
INPUT_DIR = os.path.join(STORAGE_DIR, 'inputs')
OUTPUT_DIR = os.path.join(STORAGE_DIR, 'outputs')
TEMP_DIR = os.path.join(STORAGE_DIR, 'temp')

# Create storage directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = TEMP_DIR

# Initialize inference engine - model files will be in 'models' subdirectory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.weights.h5')
STATS_PATH = os.path.join(MODEL_DIR, 'stats.npy')

print("Initializing SILICON-7 inference engine...")
inference_engine = AudioInference(MODEL_PATH, STATS_PATH)
print("Inference engine ready!")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle audio file upload and return prediction
    """
    try:
        print("Received prediction request")

        if 'audio' not in request.files:
            print("Error: No audio file in request")
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        print(f"Audio file received: {audio_file.filename}")

        if audio_file.filename == '':
            print("Error: Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Save uploaded file temporarily with unique name
        import time
        # Detect file extension from content type or filename
        ext = '.wav'
        if audio_file.content_type:
            if 'webm' in audio_file.content_type:
                ext = '.webm'
            elif 'ogg' in audio_file.content_type:
                ext = '.ogg'
            elif 'mpeg' in audio_file.content_type or 'mp3' in audio_file.content_type:
                ext = '.mp3'
        elif audio_file.filename:
            # Fallback to filename extension
            if audio_file.filename.lower().endswith('.mp3'):
                ext = '.mp3'
            elif audio_file.filename.lower().endswith('.webm'):
                ext = '.webm'
            elif audio_file.filename.lower().endswith('.ogg'):
                ext = '.ogg'

        filename = f"audio_{int(time.time())}{ext}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        print(f"Saved to: {temp_path} (type: {audio_file.content_type})")

        # Convert to WAV if needed
        wav_path = temp_path
        if ext != '.wav':
            print(f"Converting {ext} to WAV...")
            wav_path = temp_path.replace(ext, '.wav')
            try:
                import subprocess
                # Use ffmpeg to convert
                result = subprocess.run(
                    ['ffmpeg', '-i', temp_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y'],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    # Try pydub as fallback
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(temp_path)
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(wav_path, format='wav')
                        print("Converted using pydub")
                    except Exception as e:
                        print(f"Pydub conversion failed: {e}")
                        raise RuntimeError("Could not convert audio format. Please upload WAV files or install ffmpeg.")
                else:
                    print("Converted using ffmpeg")
            except FileNotFoundError:
                # ffmpeg not found, try pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(temp_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(wav_path, format='wav')
                    print("Converted using pydub")
                except Exception as e:
                    print(f"Pydub conversion failed: {e}")
                    raise RuntimeError("Could not convert audio format. Please upload WAV files or install ffmpeg/pydub.")
            finally:
                # Clean up original file
                if os.path.exists(temp_path) and temp_path != wav_path:
                    os.remove(temp_path)

        try:
            # Run inference
            print("Running inference...")
            result = inference_engine.predict_with_details(wav_path)
            print(f"Prediction result: {result}")

            # Generate unique ID for this prediction
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            prediction_id = f"pred_{timestamp}"

            # Store input WAV file
            stored_input_path = os.path.join(INPUT_DIR, f"{prediction_id}.wav")
            shutil.copy2(wav_path, stored_input_path)
            print(f"Stored input: {stored_input_path}")

            # Format response
            response = {
                'success': True,
                'prediction': {
                    'class': result['predicted_label'],
                    'confidence': round(result['confidence'] * 100, 2),
                    'probabilities': {
                        k: round(v * 100, 2)
                        for k, v in result['probabilities'].items()
                    }
                }
            }

            # Store output JSON
            output_data = {
                'id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'input_file': f"{prediction_id}.wav",
                'prediction': response['prediction']
            }
            output_path = os.path.join(OUTPUT_DIR, f"{prediction_id}.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Stored output: {output_path}")

            # Clean up temp file
            if os.path.exists(wav_path):
                os.remove(wav_path)

            print(f"Sending response: {response}")
            return jsonify(response)

        except Exception as e:
            print(f"Inference error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Clean up temp files on error
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if wav_path != temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        print(f"Request error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/info')
def info():
    """Return model information"""
    return jsonify({
        'model': 'SILICON-7 Audio Classifier',
        'classes': inference_engine.class_names,
        'version': '1.0',
        'description': '4-class audio classifier for Baby Cry, Dog, Unknown, and Unusual sounds'
    })

if __name__ == '__main__':
    # Use port 7860 for Hugging Face Spaces, fallback to 5001 for local
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
