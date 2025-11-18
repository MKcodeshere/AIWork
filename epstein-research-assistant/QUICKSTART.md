# ⚡ Quick Start (5 Minutes)

Get the Epstein Files Research Assistant running in 5 minutes!

## Prerequisites
- ✅ Python 3.11+ installed
- ✅ Gemini API key ([Get free key](https://ai.google.dev/))
- ✅ Dataset downloaded ([HuggingFace link](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K))

## Installation

### Option 1: Automated (Recommended)

**macOS/Linux:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

The script will:
1. Create virtual environment
2. Install dependencies
3. Create .env file
4. Launch the app

### Option 2: Manual

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run
streamlit run src/app.py
```

## Configuration

Edit `.env` file:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

## Add Dataset

Place your downloaded CSV in:
```
data/epstein_dataset.csv
```

## First Use

1. **Load Documents**: Start with 100 docs for testing
2. **Upload to Gemini**: One-time setup (~2 mins)
3. **Ask Questions**: Try example queries

## Example Questions

```
Who are the main individuals mentioned in the documents?
What locations appear most frequently?
Find communications about [specific topic]
Summarize documents from [date range]
```

## Troubleshooting

**Can't find CSV?**
- Ensure file is named `epstein_dataset.csv` in `data/` folder

**API Key error?**
- Check `.env` file has correct key (no quotes or spaces)

**Module not found?**
- Activate virtual environment: `source .venv/bin/activate`

## Cost

- **First 100 docs**: ~$0.015 (1.5 cents)
- **Full 25K docs**: ~$3.75 (one-time)
- **All queries**: FREE forever

## Next Steps

✅ Successfully running? Check out:
- [Full README](README.md) - Complete documentation
- [Setup Guide](SETUP_GUIDE.md) - Detailed walkthrough
- [Test Setup](test_setup.py) - Verify installation

## Support

```bash
# Test your setup
python test_setup.py

# Check dependencies
pip list | grep google-genai

# View logs
streamlit run src/app.py --logger.level=debug
```

---

**🚀 Launch command:**
```bash
streamlit run src/app.py
```

Happy investigating! 🔍
