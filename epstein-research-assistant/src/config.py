"""
Configuration management for Epstein Files Research Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Model Configuration
    MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash')

    # File Search Store Configuration
    FILE_SEARCH_STORE_NAME = os.getenv('FILE_SEARCH_STORE_NAME', 'epstein_documents_store')

    # Data Configuration
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/epstein_dataset.csv')

    # Upload Configuration
    MAX_UPLOAD_BATCH = int(os.getenv('MAX_UPLOAD_BATCH', '50'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '100'))

    # Application Settings
    APP_TITLE = os.getenv('APP_TITLE', 'Epstein Files Research Assistant')
    APP_ICON = os.getenv('APP_ICON', '🔍')

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please copy .env.example to .env and add your API key."
            )
        return True

    @classmethod
    def get_api_key(cls):
        """Get Gemini API key"""
        cls.validate()
        return cls.GEMINI_API_KEY
