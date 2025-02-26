import os

class Config:
    # Dataset paths
    RAVDESS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/RAVDESS')
    CREMA_D_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/CREMA-D')
    
    # Model paths
    AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/audio_model.pth')
    VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/video_model.pth')
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS settings
    ALLOWED_ORIGINS = [
        'http://localhost:5173',  # Vite dev server
        'http://localhost:4173'   # Vite preview
    ]

    # Feature extraction
    N_MFCC = 40
    N_MELS = 128
    WINDOW_SIZE = 0.025
    HOP_LENGTH = 0.010
    MAX_TIME_LENGTH = 300

    # Processing settings
    AUDIO_SAMPLE_RATE = 22050
    VIDEO_FRAME_RATE = 30
    VIDEO_FRAME_SIZE = (224, 224)
    
    # Training settings
    BATCH_SIZE = 32  # Increased from 16
    NUM_EPOCHS = 50  # Increased from 5
    LEARNING_RATE = 0.001
    
    # Memory management
    NUM_WORKERS = 0  # Disabled multiprocessing to avoid CUDA issues
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2