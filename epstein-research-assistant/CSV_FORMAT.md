# CSV Format Documentation

## Epstein Files Dataset Structure

The Epstein Files dataset from HuggingFace has the following structure:

### Columns

| Column Name | Type | Description |
|-------------|------|-------------|
| `filename` | string | Path or identifier to the original document/image in Google Drive |
| `text` | string | OCR-extracted text content from the document (processed with Google Tesseract) |

### Example Data

```csv
filename,text
"path/to/document1.jpg","This is the extracted text from document 1..."
"path/to/document2.jpg","This is the extracted text from document 2..."
```

### Data Characteristics

- **Total Documents**: ~25,000 pages/emails
- **File Size**: ~100MB CSV
- **Text Encoding**: UTF-8 (with fallback to latin-1, iso-8859-1)
- **Source**: House Oversight Committee release
- **OCR Tool**: Google Tesseract
- **Format**: Two-column CSV (filename, text)

### Column Mapping

The application automatically maps columns:

```python
{
    'text': 'text',      # Content column
    'path': 'filename'   # Source reference column
}
```

Supported column name patterns:

**For text content:**
- `text`, `content`, `ocr`, `body`, `document`

**For source path:**
- `path`, `source`, `drive`, `folder`, `location`, `filename`, `file`

### Data Quality Notes

1. **OCR Accuracy**: Quality varies based on original document image quality
2. **Empty Entries**: Some documents may have empty or minimal text
3. **Special Characters**: Text may contain OCR artifacts and special characters
4. **Encoding Issues**: The CSV processor tries multiple encodings automatically

### Processing Pipeline

1. **Load CSV**: Read with pandas, try multiple encodings
2. **Parse Columns**: Automatically detect text and path columns
3. **Clean Text**: Remove excessive whitespace, normalize formatting
4. **Extract Metadata**:
   - Parse filename/path for folder information
   - Categorize by content type (email, legal, travel, etc.)
   - Generate unique document IDs
5. **Export**: Create individual text files for Gemini upload

### Metadata Extraction

From the `filename` column, the app extracts:

```python
{
    'source_path': 'original/path/to/file.jpg',
    'folder': 'to',                          # Parent folder name
    'category': 'document'                    # Auto-detected category
}
```

**Category Detection** (based on filename/path):
- `email` - Contains 'email', 'eml'
- `legal` - Contains 'legal', 'court', 'deposition'
- `travel` - Contains 'flight', 'travel', 'log'
- `image` - Contains 'photo', 'image', '.jpg', '.png'
- `document` - Default for all others

### Custom CSV Format

If your CSV has different columns, you can:

1. **Rename columns** in your CSV to match expected patterns
2. **Update `src/utils.py`** to recognize your column names:

```python
# In parse_csv_columns function
text_patterns = ['text', 'content', 'your_column_name']
path_patterns = ['path', 'filename', 'your_path_column']
```

3. **Direct column specification** (for advanced users):

```python
# In src/csv_processor.py
self.column_mapping = {
    'text': 'your_text_column_name',
    'path': 'your_path_column_name'
}
```

### Data Validation

The processor validates each document:

- ✅ Text must not be empty
- ✅ Text length must be > 10 characters
- ✅ Text must be valid string type
- ✅ Filename/path must exist

Invalid documents are skipped with a count in the processing summary.

### Example Processing Output

```
Loading CSV from data/epstein_dataset.csv...
✅ Loaded with utf-8 encoding
✅ Loaded 25000 rows
Columns: ['filename', 'text']
Column mapping: {'text': 'text', 'path': 'filename'}

Processing documents from CSV...
Using 'text' for text and 'filename' for source path
100%|████████████| 25000/25000 [00:45<00:00, 550.23it/s]
✅ Processed 24873 valid documents

Document Statistics:
- Total documents: 24873
- Total characters: 125,450,320
- Avg chars per doc: 5,043
- Categories:
  - document: 18,234
  - email: 4,521
  - legal: 1,892
  - travel: 226
```

### Tips for Large CSVs

For the 100MB Epstein dataset:

1. **Start Small**: Process 100-500 docs for testing
2. **Chunked Reading**: For CSVs > 500MB, use pandas chunked reading
3. **Memory**: Ensure 4GB+ RAM available
4. **Batch Upload**: Use `MAX_UPLOAD_BATCH=50` in .env for optimal upload

### Troubleshooting CSV Issues

**"Could not identify required columns"**
- Check your CSV has 'text' or 'content' column
- Check your CSV has 'filename', 'path', or 'source' column
- Add your column names to the patterns in `utils.py`

**"UnicodeDecodeError"**
- The processor tries utf-8, latin-1, and iso-8859-1 automatically
- If still failing, convert your CSV to UTF-8 first

**"No valid documents processed"**
- Check text column isn't empty
- Verify text length > 10 characters
- Look for encoding issues

**Memory errors**
- Process in smaller batches (100-1000 docs)
- Close other applications
- Use a machine with more RAM

---

**Dataset Source**: [HuggingFace - tensonaut/EPSTEIN_FILES_20K](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)

**Last Updated**: November 2025
