"""
Gemini File Search Manager
Handles File Search Store operations and document uploads
"""
import time
from typing import List, Dict, Optional, Tuple
from google import genai
from google.genai import types
from tqdm import tqdm


class FileSearchManager:
    """Manage Gemini File Search operations"""

    def __init__(self, api_key: str, store_name: str = "epstein_documents_store"):
        """
        Initialize File Search Manager

        Args:
            api_key: Gemini API key
            store_name: Name for the File Search store
        """
        self.client = genai.Client(api_key=api_key)
        self.store_name = store_name
        self.store = None
        self.uploaded_files = {}

    def create_or_get_store(self) -> types.FileSearchStore:
        """
        Create a new File Search store or get existing one

        Returns:
            File Search Store object
        """
        print(f"\n🔍 Setting up File Search Store: {self.store_name}")

        try:
            # Try to list existing stores
            stores = self.client.file_search_stores.list()

            # Check if our store exists
            for store in stores:
                if store.display_name == self.store_name:
                    print(f"✅ Found existing store: {store.name}")
                    self.store = store
                    return self.store

            # Create new store if not found
            print(f"Creating new File Search store...")
            self.store = self.client.file_search_stores.create(
                config={'display_name': self.store_name}
            )
            print(f"✅ Created store: {self.store.name}")
            return self.store

        except Exception as e:
            raise Exception(f"Error creating/getting File Search store: {str(e)}")

    def upload_file(
        self,
        file_path: str,
        display_name: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Upload a single file to the File Search store

        Args:
            file_path: Path to the file
            display_name: Display name for the file
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (success, message)
        """
        if not self.store:
            raise ValueError("Store not initialized. Call create_or_get_store() first.")

        try:
            # Upload and import file
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=self.store.name,
                config={
                    'display_name': display_name,
                    'metadata': metadata or {}
                }
            )

            # Poll until complete
            max_wait = 300  # 5 minutes
            elapsed = 0
            poll_interval = 2

            while not operation.done and elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                operation = self.client.operations.get(operation.name)

            if operation.done:
                self.uploaded_files[display_name] = {
                    'file_path': file_path,
                    'metadata': metadata,
                    'status': 'completed'
                }
                return True, "Upload successful"
            else:
                return False, f"Upload timed out after {max_wait}s"

        except Exception as e:
            return False, f"Upload failed: {str(e)}"

    def upload_documents_batch(
        self,
        file_documents: List[Tuple[str, Dict]],
        batch_size: int = 10,
        progress_callback=None
    ) -> Dict:
        """
        Upload multiple documents in batches

        Args:
            file_documents: List of (file_path, document) tuples
            batch_size: Number of concurrent uploads
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with upload statistics
        """
        if not self.store:
            raise ValueError("Store not initialized. Call create_or_get_store() first.")

        print(f"\n📤 Uploading {len(file_documents)} documents in batches of {batch_size}...")

        stats = {
            'total': len(file_documents),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        # Process in batches
        for i in tqdm(range(0, len(file_documents), batch_size)):
            batch = file_documents[i:i + batch_size]

            for file_path, doc in batch:
                # Create display name
                display_name = f"doc_{doc['id']}"

                # Upload file
                success, message = self.upload_file(
                    file_path=file_path,
                    display_name=display_name,
                    metadata={
                        'source_path': doc['source_path'],
                        'category': doc['metadata'].get('category', 'unknown'),
                        'doc_id': doc['id']
                    }
                )

                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    stats['errors'].append({
                        'file': file_path,
                        'error': message
                    })

                if progress_callback:
                    progress_callback(stats['successful'], stats['total'])

            # Small delay between batches
            if i + batch_size < len(file_documents):
                time.sleep(1)

        print(f"\n✅ Upload complete: {stats['successful']} successful, {stats['failed']} failed")

        if stats['errors']:
            print(f"⚠️ First 5 errors:")
            for error in stats['errors'][:5]:
                print(f"  - {error['file']}: {error['error']}")

        return stats

    def list_documents(self) -> List[Dict]:
        """
        List all documents in the store

        Returns:
            List of document information
        """
        if not self.store:
            raise ValueError("Store not initialized. Call create_or_get_store() first.")

        try:
            # Note: The API might have different methods to list documents
            # This is a placeholder - adjust based on actual API
            return list(self.uploaded_files.values())

        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []

    def get_store_info(self) -> Dict:
        """
        Get information about the File Search store

        Returns:
            Dictionary with store information
        """
        if not self.store:
            return {'status': 'not_initialized'}

        return {
            'name': self.store.name,
            'display_name': self.store.display_name,
            'uploaded_count': len(self.uploaded_files),
            'status': 'active'
        }

    def delete_store(self) -> bool:
        """
        Delete the File Search store

        Returns:
            True if successful
        """
        if not self.store:
            return False

        try:
            self.client.file_search_stores.delete(self.store.name)
            print(f"✅ Deleted store: {self.store.name}")
            self.store = None
            self.uploaded_files = {}
            return True

        except Exception as e:
            print(f"Error deleting store: {str(e)}")
            return False

    def get_store(self):
        """Get the current store object"""
        return self.store
