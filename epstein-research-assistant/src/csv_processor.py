"""
CSV Processing Module for Epstein Dataset
Handles loading, parsing, and preparing documents for upload
"""
import pandas as pd
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from .utils import clean_text, extract_metadata_from_path, create_document_id, parse_csv_columns


class CSVProcessor:
    """Process Epstein dataset CSV file"""

    def __init__(self, csv_path: str):
        """
        Initialize CSV processor

        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.column_mapping = {}
        self.documents = []

    def load_csv(self) -> pd.DataFrame:
        """
        Load CSV file into DataFrame

        Returns:
            Pandas DataFrame
        """
        print(f"Loading CSV from {self.csv_path}...")

        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ Loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                raise ValueError("Could not load CSV with any encoding")

            print(f"✅ Loaded {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")

            # Identify column mapping
            self.column_mapping = parse_csv_columns(self.df)
            print(f"Column mapping: {self.column_mapping}")

            return self.df

        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def process_documents(self, limit: int = None) -> List[Dict]:
        """
        Process CSV rows into document objects

        Args:
            limit: Optional limit on number of documents to process

        Returns:
            List of document dictionaries
        """
        if self.df is None:
            raise ValueError("CSV not loaded. Call load_csv() first.")

        text_col = self.column_mapping.get('text')
        path_col = self.column_mapping.get('path')

        if not text_col or not path_col:
            raise ValueError(
                f"Could not identify required columns. "
                f"Found mapping: {self.column_mapping}. "
                f"Available columns: {list(self.df.columns)}"
            )

        print(f"\nProcessing documents from CSV...")
        print(f"Using '{text_col}' for text and '{path_col}' for source path")

        # Apply limit if specified
        df_to_process = self.df.head(limit) if limit else self.df

        self.documents = []

        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            # Extract text and path
            text = row[text_col]
            source_path = row[path_col]

            # Skip if text is missing or empty
            if pd.isna(text) or not str(text).strip():
                continue

            # Clean text
            cleaned_text = clean_text(str(text))

            if not cleaned_text or len(cleaned_text) < 10:
                continue

            # Extract metadata
            metadata = extract_metadata_from_path(str(source_path))

            # Create document object
            doc = {
                'id': create_document_id(cleaned_text, str(source_path)),
                'text': cleaned_text,
                'source_path': str(source_path),
                'metadata': metadata,
                'row_index': idx
            }

            self.documents.append(doc)

        print(f"✅ Processed {len(self.documents)} valid documents")
        return self.documents

    def get_document_stats(self) -> Dict:
        """
        Get statistics about processed documents

        Returns:
            Dictionary with statistics
        """
        if not self.documents:
            return {}

        categories = {}
        total_chars = 0

        for doc in self.documents:
            category = doc['metadata'].get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            total_chars += len(doc['text'])

        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'avg_chars_per_doc': total_chars // len(self.documents) if self.documents else 0,
            'categories': categories
        }

    def export_documents_as_txt(self, output_dir: str) -> List[Tuple[str, Dict]]:
        """
        Export documents as individual text files

        Args:
            output_dir: Directory to save text files

        Returns:
            List of (file_path, document) tuples
        """
        if not self.documents:
            raise ValueError("No documents processed. Call process_documents() first.")

        os.makedirs(output_dir, exist_ok=True)

        exported_files = []

        print(f"\nExporting documents to {output_dir}...")

        for doc in tqdm(self.documents):
            # Create filename
            filename = f"doc_{doc['id']}.txt"
            filepath = os.path.join(output_dir, filename)

            # Write content
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write(f"SOURCE: {doc['source_path']}\n")
                f.write(f"CATEGORY: {doc['metadata'].get('category', 'unknown')}\n")
                f.write(f"{'='*80}\n\n")
                f.write(doc['text'])

            exported_files.append((filepath, doc))

        print(f"✅ Exported {len(exported_files)} files")
        return exported_files

    def get_documents(self) -> List[Dict]:
        """Get processed documents"""
        return self.documents

    def get_source_mapping(self) -> Dict[str, str]:
        """
        Create mapping of document IDs to source paths

        Returns:
            Dictionary mapping doc_id -> source_path
        """
        return {doc['id']: doc['source_path'] for doc in self.documents}
