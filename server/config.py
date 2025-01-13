import os

class Config:
    # Dataset paths - update these paths
    RAVDESS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets\\RAVDESS')
    CREMA_D_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets\\CREMA-D') 
    
    # Model paths
    AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models\\audio_model.pth')
    VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models\\video_model.pth')
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS settings
    ALLOWED_ORIGINS = [
        'http://localhost:5173',  # Vite dev server
        'http://localhost:4173'   # Vite preview
    ]
    

    # Feature extraction
    N_MFCC = 13
    N_MELS = 40
    WINDOW_SIZE = 0.025
    HOP_LENGTH = 0.010
    WINDOW_SIZE = 0.025  # 25 ms
    MAX_TIME_LENGTH = 300  # Fixed length for padding/trimming

        # Processing settings
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FRAME_RATE = 30
    VIDEO_FRAME_SIZE = (112, 112)  # Reduced from 224x224 for memory efficiency
    
    # Training settings
    BATCH_SIZE = 32  # Reduced batch size
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
