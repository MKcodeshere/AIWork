# 🔧 Troubleshooting Guide

Common issues and their solutions for the Epstein Files Research Assistant.

## Import Errors

### Error: `ImportError: attempted relative import with no known parent package`

**Symptom:**
```
from .utils import clean_text, extract_metadata_from_path, create_document_id
ImportError: attempted relative import with no known parent package
```

**Solution:**
✅ **FIXED** - Update to latest code:
```bash
git pull origin claude/process-epstein-ocr-dataset-01TfiEWCuLrrHjZD3h5Docca
```

The imports have been changed from relative (`from .utils`) to absolute (`from utils`).

---

### Error: `ModuleNotFoundError: No module named 'streamlit'`

**Symptom:**
Dependencies not installed.

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration Errors

### Error: `GEMINI_API_KEY not found`

**Symptom:**
App shows "API Key not configured" error.

**Solution:**
1. Create `.env` file from template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key (no quotes):
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. Get API key from: https://ai.google.dev/

---

### Error: Path issues on Windows

**Symptom:**
Multiple nested directories like:
```
C:\epstein-research-assistant\epstein-research-assistant\epstein-research-assistant\
```

**Solution:**
Ensure you're in the correct directory:
```cmd
# Navigate to the project
cd C:\path\to\epstein-research-assistant

# Verify you see src/ folder
dir

# Run from this location
streamlit run src\app.py
```

---

## Dataset Errors

### Error: `CSV file not found`

**Symptom:**
```
❌ CSV file not found at: data/epstein_dataset.csv
```

**Solution:**
1. Download dataset from [HuggingFace](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)
2. Create `data` folder if it doesn't exist:
   ```bash
   mkdir data
   ```
3. Place CSV file as:
   ```
   data/epstein_dataset.csv
   ```

---

### Error: `UnicodeDecodeError` when loading CSV

**Symptom:**
Error reading CSV file due to encoding issues.

**Solution:**
The CSV processor tries multiple encodings automatically (utf-8, latin-1, iso-8859-1). If it still fails, check if the CSV file is corrupted or incomplete.

---

## Runtime Errors

### Error: Upload timeout

**Symptom:**
Documents upload takes too long or times out.

**Solution:**
1. Reduce batch size in `.env`:
   ```
   MAX_UPLOAD_BATCH=10
   ```

2. Start with fewer documents (100-500) for testing

3. Check your internet connection

---

### Error: API rate limit or quota exceeded

**Symptom:**
```
Error: 429 Too Many Requests
```

**Solution:**
1. Check your Gemini API quota at https://ai.google.dev/
2. Wait a few minutes and try again
3. Reduce upload batch size
4. Consider upgrading API tier if needed

---

### Error: Memory error with large dataset

**Symptom:**
Python crashes or runs out of memory when processing large CSV.

**Solution:**
1. Process in smaller batches:
   - Start with 100 documents
   - Gradually increase to 500, 1000, etc.

2. Edit `src/csv_processor.py` to use chunked reading:
   ```python
   # Read CSV in chunks
   chunksize = 1000
   for chunk in pd.read_csv(self.csv_path, chunksize=chunksize):
       # Process chunk
   ```

3. Increase system RAM or use a machine with more memory

---

## Windows-Specific Issues

### Error: Virtual environment activation fails

**Symptom:**
```
.venv\Scripts\activate : File cannot be loaded because running scripts is disabled
```

**Solution:**
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```cmd
.venv\Scripts\activate
```

---

### Error: Path separators on Windows

**Symptom:**
File paths with forward slashes don't work.

**Solution:**
The code uses `os.path.join()` which handles paths correctly on all platforms. If you're manually editing paths, use backslashes on Windows:
```
data\epstein_dataset.csv
```

---

## Streamlit Errors

### Error: Port already in use

**Symptom:**
```
Error: Port 8501 is already in use
```

**Solution:**
1. Kill existing Streamlit process:
   ```bash
   # Find process using port 8501
   lsof -ti:8501 | xargs kill -9  # macOS/Linux

   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

2. Or use a different port:
   ```bash
   streamlit run src/app.py --server.port 8502
   ```

---

### Error: Browser doesn't open automatically

**Symptom:**
Streamlit starts but browser doesn't open.

**Solution:**
Manually open your browser and go to:
```
http://localhost:8501
```

---

## Gemini API Errors

### Error: Invalid API key

**Symptom:**
```
Error: 401 Unauthorized - Invalid API key
```

**Solution:**
1. Verify your API key in `.env` is correct
2. Ensure no extra spaces or quotes around the key
3. Generate a new API key at https://ai.google.dev/
4. Update `.env` with the new key

---

### Error: File Search not available

**Symptom:**
```
Error: File Search tool not available for this model
```

**Solution:**
1. Ensure you're using a supported model:
   - `gemini-2.5-flash` ✅
   - `gemini-2.5-pro` ✅

2. Check your `.env`:
   ```
   MODEL_NAME=gemini-2.5-flash
   ```

---

## Performance Issues

### Slow query responses

**Solution:**
1. Use `gemini-2.5-flash` instead of `pro` for faster responses
2. Ensure good internet connection
3. Check if you're hitting rate limits

---

### High costs

**Solution:**
1. Remember: **Only indexing costs money** ($0.15 per 1M tokens)
2. Storage and queries are **FREE**
3. Upload documents only once
4. Don't re-upload the same documents

---

## Debugging Tips

### Enable debug mode

Run with verbose logging:
```bash
streamlit run src/app.py --logger.level=debug
```

### Test individual components

```bash
# Test configuration
python -c "from src.config import Config; Config.validate()"

# Test CSV processor
python -c "from src.csv_processor import CSVProcessor; p = CSVProcessor('data/epstein_dataset.csv'); print(p.load_csv().shape)"

# Run setup test
python test_setup.py
```

### Check dependencies

```bash
pip list | grep -E "google-genai|streamlit|pandas"
```

### Verify file structure

```bash
# Should see src/, data/, etc.
ls -la

# Check source files
ls -la src/
```

---

## Getting Help

If you're still stuck:

1. **Check the logs**: Look for error messages in the Streamlit terminal
2. **Run test setup**: `python test_setup.py`
3. **Review documentation**:
   - [README.md](README.md)
   - [SETUP_GUIDE.md](SETUP_GUIDE.md)
   - [QUICKSTART.md](QUICKSTART.md)
4. **Gemini API docs**: https://ai.google.dev/gemini-api/docs/file-search
5. **Streamlit docs**: https://docs.streamlit.io/

---

## Quick Fixes Checklist

Before asking for help, try these:

- [ ] Virtual environment activated?
- [ ] Dependencies installed? (`pip install -r requirements.txt`)
- [ ] `.env` file created and API key set?
- [ ] Dataset CSV in `data/` folder?
- [ ] Running from correct directory? (should see `src/` folder)
- [ ] Using Python 3.11+? (`python --version`)
- [ ] Latest code pulled? (`git pull`)
- [ ] Port 8501 available?
- [ ] Internet connection working?
- [ ] API key valid and has quota?

---

**Last Updated**: November 2025
**Latest Fix**: Import error resolved (absolute imports)
