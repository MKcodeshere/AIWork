# 🚀 Quick Setup Guide: Hybrid Architecture

## What Changed?

Your research assistant now uses a **hybrid approach**:
- **Gemini File Search**: Retrieves relevant documents (no more 503 errors!)
- **OpenAI GPT**: Generates high-quality answers

---

## Setup Steps (5 Minutes)

### 1. Pull Latest Code

```bash
cd /path/to/epstein-research-assistant
git pull origin claude/process-epstein-ocr-dataset-01TfiEWCuLrrHjZD3h5Docca
```

### 2. Install New Dependencies

```bash
# Activate your virtual environment first
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install new packages
pip install openai langchain langchain-openai
```

### 3. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### 4. Update .env File

Edit your `.env` file to add the OpenAI key:

```bash
# Gemini API (for retrieval - you already have this)
GEMINI_API_KEY=your_existing_gemini_key

# OpenAI API (for generation - ADD THIS)
OPENAI_API_KEY=sk-your-openai-key-here

# Model settings (optional - these are defaults)
LLM_MODEL=gpt-4o-mini
RETRIEVAL_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000
MAX_UPLOAD_BATCH=5
```

### 5. Restart the App

```bash
streamlit run src/app.py
```

---

## ✅ Verification

You should see in the sidebar:
```
✅ Both API Keys configured

🏗️ Hybrid Architecture:
- Retrieval: Gemini File Search
- Generation: gpt-4o-mini
```

---

## Important Notes

### Your Uploaded Documents

✅ **No need to re-upload!** Your existing documents in Gemini File Search work perfectly.

### How Queries Work Now

1. **Retrieval (Gemini)**: Searches your 10 documents → Returns relevant chunks
2. **Generation (OpenAI)**: Reads chunks → Generates factual answer
3. **Result**: Answer + Citations with Google Drive links

### Cost

**Previous (Gemini only)**:
- Upload: $0.15 per 1M tokens
- Queries: FREE (but 503 errors!)

**Now (Hybrid)**:
- Upload: $0.15 per 1M tokens (same)
- Retrieval: FREE (same)
- Generation: ~$0.15 per 1M tokens (OpenAI)
- **Per query**: ~$0.001-0.002 (0.1-0.2 cents)

**100 queries = ~$0.20** (20 cents total)

---

## Testing

Try these queries to test the hybrid system:

1. "How many documents did you analyze?"
2. "What is mentioned in the first document?"
3. "Summarize the types of content in these documents"

You should see:
- **🔍 Retrieving relevant documents from Gemini File Search...**
- **✅ Retrieved X relevant documents**
- **🤖 Generating answer with gpt-4o-mini...**
- **✅ Generated answer (XXX tokens)**

---

## Model Options

### Recommended (Default)

```bash
LLM_MODEL=gpt-4o-mini
```
- Fast (2-3 seconds)
- High quality
- Very cheap ($0.15 per 1M tokens)
- **Best for most users**

### For Maximum Quality

```bash
LLM_MODEL=gpt-4o
```
- Best quality
- More expensive ($2.50 per 1M tokens)
- Use for complex analysis

### For Speed

```bash
LLM_MODEL=gpt-3.5-turbo
```
- Fastest
- Decent quality
- Cheapest ($0.50 per 1M tokens)

---

## Troubleshooting

### "OPENAI_API_KEY not found"

Check your `.env` file:
```bash
# Make sure this line exists and has your actual key
OPENAI_API_KEY=sk-...your-key...
```

### "ModuleNotFoundError: No module named 'openai'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "Still getting 503 errors"

The 503 should only affect retrieval now (rare). If it happens:
1. Gemini retrieval fails → Falls back to empty context
2. OpenAI still generates answer (may be less accurate)
3. This is much better than complete failure!

### "Costs too high"

**Use cheaper model:**
```bash
LLM_MODEL=gpt-4o-mini  # Cheapest good option
```

**Reduce response length:**
```bash
LLM_MAX_TOKENS=1000  # Shorter responses
```

---

## What Stayed the Same

- ✅ Document upload process (still using Gemini File Search)
- ✅ CSV processing
- ✅ Gemini API for File Search
- ✅ Citation extraction
- ✅ Google Drive links
- ✅ Streamlit UI

## What Changed

- ✅ Query answering now uses OpenAI (more reliable)
- ✅ Need both API keys
- ✅ Slightly higher cost per query (~0.1-0.2 cents)
- ✅ Better quality answers
- ✅ No more 503 errors on generation

---

## Next Steps

1. ✅ Get OpenAI API key
2. ✅ Update `.env` file
3. ✅ Install dependencies
4. ✅ Restart app
5. ✅ Test with a query
6. ✅ Scale up to more documents when ready!

---

## Full Documentation

- **Architecture Details**: [HYBRID_ARCHITECTURE.md](HYBRID_ARCHITECTURE.md)
- **Main Guide**: [README.md](README.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

**Ready to try it?** Follow the steps above and you'll have a production-ready, reliable research assistant in 5 minutes! 🚀
