# 📖 Detailed Setup Guide

## Step-by-Step Installation

### 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Click "Get API Key"
3. Sign in with your Google account
4. Create a new API key
5. Copy the key (keep it secure!)

### 2. Download the Dataset

**Option A: Direct Download**
1. Visit [HuggingFace Dataset](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)
2. Click "Files and versions"
3. Download the CSV file
4. Place it in `epstein-research-assistant/data/epstein_dataset.csv`

**Option B: Using HuggingFace CLI**
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download dataset
huggingface-cli download tensonaut/EPSTEIN_FILES_20K --repo-type dataset --local-dir data/
```

### 3. Environment Setup

**On macOS/Linux:**
```bash
# Navigate to project
cd epstein-research-assistant

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**On Windows:**
```cmd
# Navigate to project
cd epstein-research-assistant

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env  # or use your favorite editor
```

**Required settings:**
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Optional settings (use defaults):**
```
MODEL_NAME=gemini-2.5-flash
FILE_SEARCH_STORE_NAME=epstein_documents_store
DATASET_PATH=data/epstein_dataset.csv
MAX_UPLOAD_BATCH=50
```

### 5. Verify Setup

```bash
# Check Python version (should be 3.11+)
python --version

# Check dependencies
pip list | grep google-genai
pip list | grep streamlit

# Check data file
ls -lh data/epstein_dataset.csv
```

### 6. First Run

```bash
# Start the application
streamlit run src/app.py
```

Your browser should open to `http://localhost:8501`

## First-Time Walkthrough

### 1. Load Documents (in the UI)

1. The app detects your CSV automatically
2. Start with **100 documents** for testing
3. Click "📊 Process Documents"
4. Wait for processing to complete (~10-30 seconds)

### 2. Upload to Gemini

1. Click "🚀 Upload to Gemini"
2. Files are exported to `data/processed/`
3. Upload progress shows in real-time
4. First upload takes 2-5 minutes for 100 docs

**Cost estimate for 100 docs:** ~$0.015 (1.5 cents)

### 3. Ask Your First Question

Try these starter questions:

**Simple:**
- "How many documents are in the dataset?"
- "What categories of documents are included?"

**Medium:**
- "What locations are mentioned most frequently?"
- "Who appears most often in the documents?"

**Complex:**
- "Can you summarize the types of communications found?"
- "What time periods do these documents cover?"

### 4. Understanding Results

**Answer Section:**
- AI-generated response based on documents
- Synthesized from multiple sources
- Fact-based, no speculation

**Citations Section:**
- Click "📚 View Sources" to expand
- Each citation links to original Google Drive path
- Preview text shows relevant excerpt

## Scaling to Full Dataset

Once comfortable with 100 documents:

### 1. Process More Documents

In the UI:
1. Reset the app (sidebar: "🔄 Reset Application")
2. Choose 1,000 documents
3. Process and upload
4. Test queries

### 2. Eventually Process All 25K

**Recommended approach:**
- 100 docs → Test features
- 1,000 docs → Test performance
- 5,000 docs → Test real queries
- 25,000 docs → Full production

**Time estimates:**
- 1,000 docs: ~15 min upload
- 5,000 docs: ~1 hour upload
- 25,000 docs: ~4-5 hours upload

**Cost for 25K docs:** ~$3.75 one-time

## Common Issues & Solutions

### Issue: "CSV file not found"
**Solution:**
```bash
# Check file exists
ls -la data/epstein_dataset.csv

# If missing, check your download
# Ensure filename matches exactly
```

### Issue: "API Key not configured"
**Solution:**
```bash
# Check .env file exists
cat .env | grep GEMINI_API_KEY

# Verify no extra spaces or quotes
# Should be: GEMINI_API_KEY=actual_key_here
# NOT: GEMINI_API_KEY="actual_key_here"
```

### Issue: "Module not found"
**Solution:**
```bash
# Ensure virtual environment is activated
# You should see (.venv) in your terminal

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Upload timeout"
**Solution:**
Edit `.env`:
```
MAX_UPLOAD_BATCH=10  # Reduce from 50
```

### Issue: "Memory error with large dataset"
**Solution:**
Process in smaller batches:
- Start with 100-500 documents
- Upload in multiple sessions
- Or increase system RAM

## Performance Tips

### For Faster Queries
- Use `gemini-2.5-flash` (default)
- Enable caching in sidebar

### For Better Accuracy
- Use `gemini-2.5-pro`
- Ask specific questions
- Reference document categories

### For Cost Optimization
- Upload only once (storage is free)
- Queries are free
- Only indexing costs money

## Advanced Configuration

### Custom CSV Columns

If your CSV has different column names, edit `src/utils.py`:

```python
def parse_csv_columns(df):
    # Add your custom column patterns
    text_patterns = ['text', 'content', 'your_text_column']
    path_patterns = ['path', 'source', 'your_path_column']
```

### Custom Prompts

Edit `src/query_engine.py` to customize AI behavior:

```python
system_instruction = """
You are a specialized investigative assistant.

Your custom instructions here...
"""
```

### Multiple Datasets

To work with multiple datasets:
1. Use different `FILE_SEARCH_STORE_NAME` in `.env`
2. Each dataset gets its own search store
3. Switch between them in config

## Next Steps

✅ Setup complete? Try these:
1. Ask 10 different questions
2. Explore citation sources
3. Test different query types
4. Scale up to more documents
5. Customize for your research needs

## Getting Help

**Documentation:**
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Streamlit Docs](https://docs.streamlit.io/)
- [File Search Guide](https://ai.google.dev/gemini-api/docs/file-search)

**Debugging:**
```bash
# Check logs
streamlit run src/app.py --logger.level=debug

# Test individual components
python -c "from src.config import Config; Config.validate()"
```

---

**Ready to investigate? Launch the app:**
```bash
streamlit run src/app.py
```
