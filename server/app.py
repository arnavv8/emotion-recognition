from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tempfile
import os
import cv2
import numpy as np
from config import Config
from models.audio_model import AudioEmotionModel
from models.video_model import VideoEmotionModel
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor

app = Flask(__name__)
CORS(app, origins=Config.ALLOWED_ORIGINS)

# Load models
try:
    audio_model = AudioEmotionModel()
    audio_model.load_state_dict(torch.load(Config.AUDIO_MODEL_PATH))
    audio_model.eval()

    video_model = VideoEmotionModel()
    video_model.load_state_dict(torch.load(Config.VIDEO_MODEL_PATH))
    video_model.eval()
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Please ensure models are trained before running the server.")
    exit(1)

# Load processors
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

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
        
        if media_type not in ['audio', 'video']:
            return jsonify({'error': 'Invalid media type'}), 400
        
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            if media_type == 'audio':
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
            else:
                # Process video
                frames = video_processor.extract_frames(temp_path)
                with torch.no_grad():
                    output = video_model(frames)
                confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
                
                return jsonify({
                    'emotion': EMOTIONS[prediction.item()],
                    'confidence': float(confidence.item()),
                    'videoScore': float(confidence.item())
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

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        
        # Validate frame
        is_valid, error = validate_file(frame_file)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Process frame
        frame_tensor = video_processor.process_frame(frame)
        
        with torch.no_grad():
            output = video_model(frame_tensor)
        confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
        
        return jsonify({
            'emotion': EMOTIONS[prediction.item()],
            'confidence': float(confidence.item()),
            'videoScore': float(confidence.item())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT)