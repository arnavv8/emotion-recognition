import os

class Config:
    # Dataset paths
    RAVDESS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/RAVDESS')
    CREMA_D_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/CREMA-D')
    
    # Model paths - dataset-specific models
    AUDIO_MODEL_RAVDESS_PATH = os.path.join(os.path.dirname(__file__), 'models/audio_model_ravdess.pth')
    AUDIO_MODEL_CREMAD_PATH = os.path.join(os.path.dirname(__file__), 'models/audio_model_cremad.pth')
    VIDEO_MODEL_CREMAD_PATH = os.path.join(os.path.dirname(__file__), 'models/video_model_cremad.pth')
    
    # Default model selection (can be changed via environment variable)
    ACTIVE_DATASET = os.environ.get('EMOTION_DATASET', 'ravdess')  # 'ravdess' or 'cremad'
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS settings
    ALLOWED_ORIGINS = [
        'http://localhost:5173',  # Vite dev server
        'http://localhost:4173'   # Vite preview
    ]

    # Feature extraction parameters
    AUDIO_SAMPLE_RATE = 22050
    N_MFCC = 40
    N_MELS = 128
    N_FFT = 1024
    WINDOW_SIZE = 0.025  # in seconds
    HOP_LENGTH = 0.010   # in seconds
    MAX_TIME_LENGTH = 200  # Maximum number of time steps
    
    # Augmentation settings
    AUGMENTATION_ENABLED = True
    PITCH_SHIFT_RANGE = (-3, 3)
    TIME_STRETCH_RANGE = (0.8, 1.2)
    NOISE_FACTOR = 0.01
    
    # Video settings
    VIDEO_FRAME_RATE = 30
    VIDEO_FRAME_SIZE = (224, 224)
    
    # Training settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 20
    
    # Memory management
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Gradient clipping
    GRADIENT_CLIP_VALUE = 5.0
    CLASS_WEIGHTS_ENABLED = True
    
    # Learning rate scheduler settings
    LR_SCHEDULER_FACTOR = 0.1
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_MIN_LR = 1e-6
    
    # Dataset-specific settings
    RAVDESS_SETTINGS = {
        'N_MFCC': 40,
        'N_MELS': 128,
        'AUGMENTATION_STRENGTH': 1.2
    }
    
    CREMAD_SETTINGS = {
        'N_MFCC': 40,
        'N_MELS': 128,
        'AUGMENTATION_STRENGTH': 1.0
    }
    
    @classmethod
    def set_active_dataset(cls, dataset):
        """Set the active dataset and update related configurations."""
        if dataset not in ['ravdess', 'cremad']:
            raise ValueError("Dataset must be 'ravdess' or 'cremad'")
        
        cls.ACTIVE_DATASET = dataset
        os.environ['EMOTION_DATASET'] = dataset
        
        # Update dataset-specific configurations
        if dataset == 'ravdess':
            cls.N_MFCC = cls.RAVDESS_SETTINGS['N_MFCC']
            cls.N_MELS = cls.RAVDESS_SETTINGS['N_MELS']
        else:  # cremad
            cls.N_MFCC = cls.CREMAD_SETTINGS['N_MFCC']
            cls.N_MELS = cls.CREMAD_SETTINGS['N_MELS']
        
        return cls.ACTIVE_DATASET