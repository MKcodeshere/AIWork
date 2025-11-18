"""
Utility functions for Epstein Files Research Assistant
"""
import hashlib
import re
from typing import Dict, Any, List


def create_document_id(text: str, source_path: str) -> str:
    """
    Create a unique document ID based on content and source

    Args:
        text: Document text content
        source_path: Google Drive path

    Returns:
        Unique document ID (hash)
    """
    content = f"{text[:100]}{source_path}"
    return hashlib.md5(content.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean OCR text for better processing

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might interfere
    text = text.strip()

    return text


def extract_metadata_from_path(path: str) -> Dict[str, str]:
    """
    Extract metadata from Google Drive path or filename

    Args:
        path: Google Drive folder path or filename

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'source_path': path,
        'folder': '',
        'category': ''
    }

    # Handle both forward slashes and backslashes
    if '/' in path or '\\' in path:
        # Normalize path separators
        normalized_path = path.replace('\\', '/')
        parts = normalized_path.split('/')
        metadata['folder'] = parts[-2] if len(parts) > 1 else ''  # Get parent folder

        # Extract filename
        filename = parts[-1] if parts else path
    else:
        # Just a filename, no path
        filename = path
        metadata['folder'] = ''

    # Try to identify category from path or filename
    path_lower = path.lower()
    if 'email' in path_lower or 'eml' in path_lower:
        metadata['category'] = 'email'
    elif 'legal' in path_lower or 'court' in path_lower or 'deposition' in path_lower:
        metadata['category'] = 'legal'
    elif 'flight' in path_lower or 'travel' in path_lower or 'log' in path_lower:
        metadata['category'] = 'travel'
    elif 'photo' in path_lower or 'image' in path_lower or '.jpg' in path_lower or '.png' in path_lower:
        metadata['category'] = 'image'
    else:
        metadata['category'] = 'document'

    return metadata


def format_citation(citation: Dict[str, Any], source_mapping: Dict[str, str]) -> str:
    """
    Format citation with Google Drive link

    Args:
        citation: Citation data from Gemini response
        source_mapping: Mapping of document IDs to Google Drive paths

    Returns:
        Formatted citation string
    """
    # Extract source information
    source_id = citation.get('source_id', 'unknown')

    # Get Google Drive link if available
    drive_path = source_mapping.get(source_id, '')

    if drive_path:
        return f"📄 Source: [{drive_path}]({drive_path})"
    else:
        return f"📄 Source: Document {source_id}"


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to max length with ellipsis

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def parse_csv_columns(df) -> Dict[str, str]:
    """
    Identify relevant columns in the CSV

    Args:
        df: Pandas DataFrame

    Returns:
        Dictionary mapping expected fields to actual column names
    """
    columns = df.columns.tolist()
    column_mapping = {}

    # Common patterns for text content
    text_patterns = ['text', 'content', 'ocr', 'body', 'document']
    # Common patterns for source path (including 'filename' for Epstein dataset)
    path_patterns = ['path', 'source', 'drive', 'folder', 'location', 'filename', 'file']

    for col in columns:
        col_lower = col.lower()

        # Find text column
        if not column_mapping.get('text'):
            for pattern in text_patterns:
                if pattern in col_lower:
                    column_mapping['text'] = col
                    break

        # Find path column
        if not column_mapping.get('path'):
            for pattern in path_patterns:
                if pattern in col_lower:
                    column_mapping['path'] = col
                    break

    return column_mapping


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
