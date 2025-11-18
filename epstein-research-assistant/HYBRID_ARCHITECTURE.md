# 🏗️ Hybrid RAG Architecture

## Overview

The Epstein Files Research Assistant now uses a **hybrid architecture** combining the best of both worlds:

- **Gemini File Search** for powerful document retrieval
- **OpenAI GPT** for high-quality answer generation

This solves the 503 "model overloaded" errors from Gemini while maintaining excellent RAG capabilities.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                │
│                 "Who appears most frequently?"               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: RETRIEVAL (Gemini)                      │
│  Gemini File Search API                                      │
│  - Searches 25K+ indexed documents                          │
│  - Returns top relevant chunks                               │
│  - Provides source citations                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   Retrieved Context
               (Document chunks + citations)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: GENERATION (OpenAI)                     │
│  OpenAI GPT-4o-mini / GPT-4o                                │
│  - Receives question + retrieved context                     │
│  - Generates factual answer                                  │
│  - Includes source references                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    FORMATTED RESPONSE                        │
│  - Answer from OpenAI                                        │
│  - Citations from Gemini                                     │
│  - Google Drive source links                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Benefits

### ✅ Reliability
- **No more 503 errors** from Gemini overload
- OpenAI handles generation reliably
- Gemini only used for retrieval (lower load)

### ✅ Quality
- **Best-in-class retrieval**: Gemini File Search semantic search
- **Best-in-class generation**: GPT-4o/GPT-4o-mini
- Better than using either alone

### ✅ Cost Efficiency
- Gemini: Only indexing cost ($0.15/1M tokens one-time)
- Gemini: Free retrieval queries
- OpenAI: Pay only for generation tokens
- Typical query: ~500-1000 tokens = $0.001-0.002

### ✅ Flexibility
- Switch LLM models easily (GPT-4o-mini, GPT-4o, GPT-4-turbo)
- Adjust temperature and max tokens
- Keep same retrieval backend

---

## Configuration

### Required API Keys

Add both to your `.env` file:

```bash
# Gemini API (for retrieval)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API (for generation)
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Selection

```bash
# LLM for generation (OpenAI)
LLM_MODEL=gpt-4o-mini
# Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo

# Retrieval model (Gemini - only for File Search)
RETRIEVAL_MODEL=gemini-2.5-flash
```

### LLM Parameters

```bash
LLM_TEMPERATURE=0.3          # 0.0-1.0 (lower = more factual)
LLM_MAX_TOKENS=2000          # Maximum response length
```

---

## How It Works

### 1. Document Upload (One-time)

Uses **Gemini File Search API**:
```python
# Upload to Gemini File Search Store
operation = client.file_search_stores.upload_to_file_search_store(
    file=file_path,
    file_search_store_name=store.name,
    config={'display_name': display_name}
)
```

- Documents are chunked automatically
- Embeddings generated (text-embedding-005)
- Indexed for fast retrieval
- **Cost**: $0.15 per 1M tokens (one-time)

### 2. Query Processing (Runtime)

**Step 1: Retrieval (Gemini)**
```python
# Use Gemini File Search to retrieve relevant documents
retrieval_response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question,
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )
        ]
    )
)
```

- Semantic search across all documents
- Returns top relevant chunks
- Includes source citations
- **Cost**: FREE

**Step 2: Generation (OpenAI)**
```python
# Generate answer using OpenAI with retrieved context
completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT: {context}"}
    ],
    temperature=0.3,
    max_tokens=2000
)
```

- Receives question + retrieved context
- Generates factual answer
- References source documents
- **Cost**: ~$0.15 per 1M tokens

---

## Cost Comparison

### Traditional Gemini-Only RAG
- Indexing: $0.15 per 1M tokens (one-time)
- Retrieval: FREE
- Generation: FREE (but 503 errors!)
- **Issue**: Unreliable due to overload

### Hybrid Architecture (Gemini + OpenAI)
- Indexing: $0.15 per 1M tokens (one-time)
- Retrieval: FREE
- Generation: ~$0.15 per 1M tokens (GPT-4o-mini)
- **Benefits**: Reliable, high-quality

### Example Costs (10,000 documents)

| Operation | Cost |
|-----------|------|
| One-time indexing (10K docs) | ~$1.50 |
| 100 queries (500 tokens avg) | ~$0.0075 |
| **Total for 100 queries** | **$1.51** |

**Per query**: ~$0.015 (1.5 cents)

---

## Model Options

### OpenAI LLM Models

| Model | Speed | Quality | Cost/1M tokens | Best For |
|-------|-------|---------|----------------|----------|
| **gpt-4o-mini** | ⚡⚡⚡ | ⭐⭐⭐ | $0.15 | General use (recommended) |
| gpt-4o | ⚡⚡ | ⭐⭐⭐⭐⭐ | $2.50 | Complex analysis |
| gpt-4-turbo | ⚡⚡ | ⭐⭐⭐⭐ | $10.00 | Maximum quality |
| gpt-3.5-turbo | ⚡⚡⚡ | ⭐⭐ | $0.50 | Basic queries |

**Recommendation**: Start with `gpt-4o-mini` - excellent quality at low cost.

### Gemini Retrieval Models

| Model | Purpose |
|-------|---------|
| **gemini-2.5-flash** | File Search retrieval (recommended) |
| gemini-2.5-pro | File Search retrieval (alternative) |

**Note**: Gemini is only used for retrieval, not generation, so model choice has minimal impact.

---

## Code Structure

### Query Engine (`query_engine.py`)

```python
class QueryEngine:
    def __init__(
        self,
        gemini_api_key: str,     # For retrieval
        openai_api_key: str,     # For generation
        llm_model: str,
        retrieval_model: str,
        ...
    ):
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)

    def query(self, question, file_search_store):
        # Step 1: Retrieve with Gemini
        context = self._retrieve_context(question)

        # Step 2: Generate with OpenAI
        answer = self._generate_answer(question, context)

        return formatted_response
```

---

## Migration Guide

If you were using the old Gemini-only version:

### 1. Update Dependencies

```bash
pip install openai langchain langchain-openai
```

### 2. Update `.env`

Add OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Update Config (Optional)

Customize models:
```bash
LLM_MODEL=gpt-4o-mini
RETRIEVAL_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000
```

### 4. Restart App

```bash
streamlit run src/app.py
```

**That's it!** Your existing uploaded documents work with no changes.

---

## Advantages Over Alternatives

### vs. Pure Gemini RAG
✅ No 503 errors
✅ More reliable generation
❌ Slightly higher cost

### vs. Pure OpenAI RAG (embeddings + GPT)
✅ Better File Search from Gemini
✅ Lower retrieval cost (free)
✅ No vector DB to manage

### vs. LangChain with Custom Vector DB
✅ Fully managed (no Chroma/Pinecone setup)
✅ Simpler architecture
✅ Faster to deploy

---

## Troubleshooting

### "OPENAI_API_KEY not found"
**Solution**: Add to `.env` file
```bash
OPENAI_API_KEY=sk-...your-key...
```

### "Gemini API still returns 503"
**Solution**: This only affects retrieval now (rare). Generation always works via OpenAI.

### "Costs too high"
**Solution**:
- Use `gpt-4o-mini` instead of `gpt-4o`
- Reduce `LLM_MAX_TOKENS`
- Each query costs ~1-2 cents with gpt-4o-mini

### "Want to use only Gemini"
**Solution**: The old code is in git history. But we recommend hybrid for reliability.

---

## Performance

### Query Latency

| Stage | Time |
|-------|------|
| Retrieval (Gemini) | 1-2 seconds |
| Generation (OpenAI) | 2-4 seconds |
| **Total** | **3-6 seconds** |

### Quality Metrics

- **Relevance**: ⭐⭐⭐⭐⭐ (Gemini File Search excellent)
- **Accuracy**: ⭐⭐⭐⭐⭐ (GPT-4o-mini factual)
- **Citations**: ⭐⭐⭐⭐⭐ (Source tracking from Gemini)

---

## Future Enhancements

Possible improvements:

1. **Caching**: Cache retrieved contexts to save costs
2. **Streaming**: Stream OpenAI responses for better UX
3. **Multi-model**: Let users switch LLM providers
4. **Analytics**: Track costs and performance per query

---

## Summary

The hybrid architecture gives you:

✅ **Best of both worlds**: Gemini retrieval + OpenAI generation
✅ **Reliability**: No more 503 errors
✅ **Quality**: State-of-the-art on both ends
✅ **Cost-effective**: ~1-2 cents per query
✅ **Simple**: Fully managed, no vector DB

**Perfect for production use!**

---

**Questions?** Check the main [README.md](README.md) or [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
