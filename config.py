import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # App settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    # Session settings
    SESSION_TYPE = 'filesystem'
    SESSION_COOKIE_NAME = 'ai_ecom_session'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'False') == 'True'
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour in seconds
    
    # MongoDB settings
    MONGODB_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    MONGODB_NAME = os.getenv("MONGODB_DB_NAME", "ai_ecom_chatbot")
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour in seconds
    
    # Qdrant settings
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
    
    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    @classmethod
    def init_app(cls, app):
        # Ensure upload folder exists
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        
        # Update app config
        for key in dir(cls):
            if key.isupper() and not key.startswith('_'):
                app.config[key] = getattr(cls, key)
