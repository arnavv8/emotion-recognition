from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tempfile
import os
from config import Config
from models.audio_model import AudioEmotionModel
from utils.audio_processor import AudioProcessor

app = Flask(__name__)
CORS(app, origins=Config.ALLOWED_ORIGINS)

# Load audio model
try:
    audio_model = AudioEmotionModel()
    audio_model.load_state_dict(torch.load(Config.AUDIO_MODEL_PATH))
    audio_model.eval()
except Exception as e:
    print(f"Error loading audio model: {str(e)}")
    print("Please ensure the model is trained before running the server.")
    exit(1)

# Load audio processor
audio_processor = AudioProcessor()

# Emotion labels
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def validate_file(file):
    if not file:
        return False, "No file provided"
    if not file.filename:
        return False, "Invalid file"
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        media_type = request.form.get('type', '')
        
        # Validate file
        is_valid, error = validate_file(file)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        if media_type != 'audio':
            return jsonify({'error': 'Only audio processing is enabled'}), 400
        
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Process audio
            features = audio_processor.preprocess_audio(temp_path)
            with torch.no_grad():
                output = audio_model(features)
            confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
            
            return jsonify({
                'emotion': EMOTIONS[prediction.item()],
                'confidence': float(confidence.item()),
                'audioScore': float(confidence.item())
            })
        finally:
            # Clean up
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Commented out video processing
# @app.route('/predict-frame', methods=['POST'])
# def predict_frame():
#     return jsonify({'error': 'Video processing is disabled'}), 400

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT)
