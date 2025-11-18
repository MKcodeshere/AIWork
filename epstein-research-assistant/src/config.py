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
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Model Configuration
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')  # OpenAI model for generation
    RETRIEVAL_MODEL = os.getenv('RETRIEVAL_MODEL', 'gemini-2.5-flash')  # Gemini for retrieval

    # Legacy support
    MODEL_NAME = os.getenv('MODEL_NAME', LLM_MODEL)

    # File Search Store Configuration
    FILE_SEARCH_STORE_NAME = os.getenv('FILE_SEARCH_STORE_NAME', 'epstein_documents_store')

    # Data Configuration
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/epstein_dataset.csv')

    # Upload Configuration
    MAX_UPLOAD_BATCH = int(os.getenv('MAX_UPLOAD_BATCH', '5'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '100'))

    # Application Settings
    APP_TITLE = os.getenv('APP_TITLE', 'Epstein Files Research Assistant')
    APP_ICON = os.getenv('APP_ICON', '🔍')

    # LLM Settings
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '2000'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []

        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY not found (needed for RAG retrieval)")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not found (needed for LLM generation)")

        if errors:
            raise ValueError(
                "Missing required configuration:\n" +
                "\n".join(f"  - {e}" for e in errors) +
                "\n\nPlease copy .env.example to .env and add your API keys."
            )
        return True

    @classmethod
    def get_gemini_api_key(cls):
        """Get Gemini API key"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        return cls.GEMINI_API_KEY

    @classmethod
    def get_openai_api_key(cls):
        """Get OpenAI API key"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        return cls.OPENAI_API_KEY

    # Legacy method for backward compatibility
    @classmethod
    def get_api_key(cls):
        """Get Gemini API key (legacy)"""
        return cls.get_gemini_api_key()
