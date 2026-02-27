# Configuration settings for Face Recognition Application

# Camera settings
CAMERA_SETTINGS = {
    'resolution': (1920, 1080),
    'fps': 30,
    'camera_index': 0
}

# Face recognition settings
FACE_RECOGNITION_SETTINGS = {
    'model_path': 'path/to/face_recognition_model',
    'tolerance': 0.6,
    'number_of_jitters': 10
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'threads': 4,
    'enable_cuda': True,
    'log_level': 'INFO'
}

# Database settings
DATABASE_SETTINGS = {
    'db_type': 'sqlite',
    'db_name': 'face_recognition.db',
    'connection_timeout': 30
}