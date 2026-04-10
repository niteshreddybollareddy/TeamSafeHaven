"""
SILICON-9 LiveInference Web Application
Flask backend for real-time baby cry detection
"""

import os
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import subprocess

from inference import BabyCryClassifier, get_config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'storage/inputs'
app.config['OUTPUT_FOLDER'] = 'storage/outputs'
app.config['TEMP_FOLDER'] = 'storage/temp'

# Allowed audio formats
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm', 'm4a', 'flac'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Initialize classifier
config = get_config()
classifier = BabyCryClassifier(
    model_path='models/crynet_micro_v2.tflite',
    config=config
)

print("\n" + "="*70)
print("SILICON-9 LIVEINFERENCE SERVER STARTED")
print("="*70)
print(f"Model: CryNet-Micro-v2 (Binary Baby Cry Detection)")
print(f"Classes: {config['class_names']}")
print(f"Ready to accept requests...")
print("="*70 + "\n")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(input_path, output_path):
    """
    Convert audio file to 16kHz mono WAV using ffmpeg

    Args:
        input_path: Input audio file path
        output_path: Output WAV file path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use ffmpeg to convert
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-y',            # Overwrite output
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False

    except Exception as e:
        print(f"Conversion error: {e}")
        return False


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle audio file upload and run inference

    Returns:
        JSON response with prediction results
    """
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file format. Allowed: {ALLOWED_EXTENSIONS}'}), 400

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        original_filename = secure_filename(f"{timestamp}_original.{original_ext}")
        wav_filename = secure_filename(f"{timestamp}.wav")

        # Save original file
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # Convert to WAV if needed
        if original_ext != 'wav':
            wav_path = os.path.join(app.config['TEMP_FOLDER'], wav_filename)
            success = convert_to_wav(original_path, wav_path)

            if not success:
                return jsonify({'error': 'Failed to convert audio file'}), 500

            # Use converted file for inference
            inference_path = wav_path
        else:
            # Already WAV, use directly
            inference_path = original_path

        # Run inference
        result = classifier.predict(inference_path)

        # Format response to match SILICON-7 frontend expectations
        response = {
            'success': True,
            'prediction': {
                'class': result['predicted_class'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            },
            'timestamp': timestamp,
            'model': 'CryNet-Micro-v2'
        }

        # Save result to output folder
        output_filename = f"{timestamp}_result.json"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)

        print(f"[{timestamp}] Prediction: {result['predicted_class']} ({result['confidence']:.2f}%)")

        return jsonify(response), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'CryNet-Micro-v2',
        'classes': config['class_names'],
        'version': '1.0.0'
    }), 200


@app.route('/history', methods=['GET'])
def history():
    """
    Get prediction history

    Returns:
        JSON array of past predictions (last 50)
    """
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        result_files = sorted(output_dir.glob('*_result.json'), reverse=True)[:50]

        history = []
        for result_file in result_files:
            with open(result_file, 'r') as f:
                history.append(json.load(f))

        return jsonify(history), 200

    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask server
    # Use port 7860 for Hugging Face Spaces compatibility
    app.run(host='0.0.0.0', port=7860, debug=False)
