from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tempfile
import os
import cv2
import numpy as np
import json
from config import Config
from models.audio_model import AudioEmotionModel
from models.video_model import VideoEmotionModel
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor

app = Flask(__name__)
CORS(app, origins=Config.ALLOWED_ORIGINS)

# Determine which models to load based on configuration
active_dataset = Config.ACTIVE_DATASET.lower()
print(f"Active dataset: {active_dataset}")

# Standardized emotion labels for API
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Dataset-specific emotion mappings
RAVDESS_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
CREMAD_EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad']

# Map dataset emotions to standardized API emotions
def map_emotion(dataset, emotion_idx):
    if dataset == 'ravdess':
        emotion = RAVDESS_EMOTIONS[emotion_idx]
        # Map RAVDESS emotions to API emotions
        mapping = {
            'neutral': 'neutral',
            'calm': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'surprised': 'surprised'
        }
        return mapping.get(emotion, 'neutral')
    else:  # cremad
        emotion = CREMAD_EMOTIONS[emotion_idx]
        return emotion

# Load models
audio_model_ravdess = None
audio_model_cremad = None
video_model_cremad = None
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

# Load model metrics if available
metrics_file = os.path.join(os.path.dirname(__file__), 'metrics', 'model_metrics.json')
model_metrics = {}
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            model_metrics = json.load(f)
        print("Loaded model metrics:", model_metrics.keys())
    except Exception as e:
        print(f"Error loading model metrics: {str(e)}")

try:
    # Load RAVDESS audio model
    if os.path.exists(Config.AUDIO_MODEL_RAVDESS_PATH):
        audio_model_ravdess = AudioEmotionModel(num_emotions=8)  # RAVDESS has 8 emotions
        audio_model_ravdess.load_state_dict(torch.load(Config.AUDIO_MODEL_RAVDESS_PATH, map_location='cpu'))
        audio_model_ravdess.eval()
        print("RAVDESS audio model loaded successfully")
    else:
        print(f"RAVDESS audio model not found at {Config.AUDIO_MODEL_RAVDESS_PATH}")
    
    # Load CREMA-D audio model
    if os.path.exists(Config.AUDIO_MODEL_CREMAD_PATH):
        audio_model_cremad = AudioEmotionModel(num_emotions=6)  # CREMA-D has 6 emotions
        audio_model_cremad.load_state_dict(torch.load(Config.AUDIO_MODEL_CREMAD_PATH, map_location='cpu'))
        audio_model_cremad.eval()
        print("CREMA-D audio model loaded successfully")
    else:
        print(f"CREMA-D audio model not found at {Config.AUDIO_MODEL_CREMAD_PATH}")

    # Load CREMA-D video model
    if os.path.exists(Config.VIDEO_MODEL_CREMAD_PATH):
        video_model_cremad = VideoEmotionModel(num_emotions=6)  # CREMA-D has 6 emotions
        video_model_cremad.load_state_dict(torch.load(Config.VIDEO_MODEL_CREMAD_PATH, map_location='cpu'))
        video_model_cremad.eval()
        print("CREMA-D video model loaded successfully")
    else:
        print(f"CREMA-D video model not found at {Config.VIDEO_MODEL_CREMAD_PATH}")
    
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Models will be loaded on demand if available")

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def validate_file(file):
    if not file:
        return False, "No file provided"
    if not file.filename:
        return False, "Invalid file"
    return True, None

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        media_type = request.form.get('type', '')
        dataset = request.form.get('dataset', active_dataset)
        
        is_valid, error = validate_file(file)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            if media_type == 'audio':
                # Determine which audio model to use
                if dataset == 'ravdess':
                    if audio_model_ravdess is None:
                        return jsonify({'error': 'RAVDESS audio model not available'}), 503
                    audio_model = audio_model_ravdess
                    num_emotions = 8
                else:  # default to cremad
                    if audio_model_cremad is None:
                        return jsonify({'error': 'CREMA-D audio model not available'}), 503
                    audio_model = audio_model_cremad
                    num_emotions = 6
                    dataset = 'cremad'
                
                # Process audio
                features = audio_processor.preprocess_audio(temp_path, dataset_type=dataset)
                with torch.no_grad():
                    output = audio_model(features)
                confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
                
                # Map to standardized emotion
                emotion = map_emotion(dataset, prediction.item())
                
                return jsonify({
                    'emotion': emotion,
                    'confidence': float(confidence.item()),
                    'audioScore': float(confidence.item()),
                    'dataset': dataset
                })
                
            elif media_type == 'video':
                # Check if video model is available
                if video_model_cremad is None:
                    return jsonify({'error': 'Video model not available'}), 503
                
                # Process video
                frames = video_processor.extract_frames(temp_path)
                with torch.no_grad():
                    output = video_model_cremad(frames)
                confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
                
                # Map to standardized emotion
                emotion = map_emotion('cremad', prediction.item())
                
                return jsonify({
                    'emotion': emotion,
                    'confidence': float(confidence.item()),
                    'videoScore': float(confidence.item()),
                    'dataset': 'cremad'
                })
            else:
                return jsonify({'error': 'Invalid media type'}), 400
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
        
        # Check if video model is available
        if video_model_cremad is None:
            return jsonify({'error': 'Video model not available'}), 503
        
        # Convert frame data to OpenCV format
        frame_array = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'videoScore': 0.0,
                'faces': []
            })
        
        # Process the largest face
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_frame = frame[y:y+h, x:x+w]
        
        # Process face for emotion recognition
        processed_frame = video_processor.process_frame(face_frame)
        
        with torch.no_grad():
            output = video_model_cremad(processed_frame)
            confidence, prediction = torch.max(torch.softmax(output, dim=1), dim=1)
        
        # Map to standardized emotion
        emotion = map_emotion('cremad', prediction.item())
        
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence.item()),
            'videoScore': float(confidence.item()),
            'faces': [[int(x), int(y), int(w), int(h)]],
            'dataset': 'cremad'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Return the status of the models and active dataset."""
    available_models = {
        'ravdess_audio': audio_model_ravdess is not None,
        'cremad_audio': audio_model_cremad is not None,
        'cremad_video': video_model_cremad is not None
    }
    
    # Get model metrics if available
    model_performance = {}
    for model_key, is_available in available_models.items():
        if is_available and model_key in model_metrics:
            model_performance[model_key] = model_metrics[model_key]
    
    return jsonify({
        'status': 'ok',
        'activeDataset': active_dataset,
        'availableModels': available_models,
        'modelPerformance': model_performance,
        'supportedEmotions': EMOTIONS
    })

@app.route('/switch-dataset', methods=['POST'])
def switch_dataset():
    """Switch the active dataset."""
    data = request.json
    if not data or 'dataset' not in data:
        return jsonify({'error': 'Dataset not specified'}), 400
    
    new_dataset = data['dataset'].lower()
    if new_dataset not in ['ravdess', 'cremad']:
        return jsonify({'error': 'Invalid dataset. Must be "ravdess" or "cremad"'}), 400
    
    # Check if models for the requested dataset are available
    if new_dataset == 'ravdess' and audio_model_ravdess is None:
        return jsonify({'error': 'RAVDESS audio model not available'}), 503
    
    if new_dataset == 'cremad' and audio_model_cremad is None:
        return jsonify({'error': 'CREMA-D audio model not available'}), 503
    
    # Update active dataset
    global active_dataset
    active_dataset = new_dataset
    Config.set_active_dataset(new_dataset)
    
    return jsonify({
        'status': 'ok',
        'activeDataset': active_dataset,
        'message': f'Switched to {active_dataset.upper()} dataset'
    })

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=True)