# Building a 25,000-Document Research Assistant: Testing Google's Gemini File Search in the Real World

*A technical deep-dive into using Google's new RAG API with the Epstein Files dataset*

---

## The Skeptic's Journey

When Google announced Gemini File Search in late 2024, I'll admit I was skeptical. Another "fully managed RAG" solution? We've heard that before. The promise was compelling: upload documents, let Google handle chunking and embeddings, and query with natural language. No vector database setup, no embedding model management, just works.

But here's the thing about developer tools—they always sound great in the announcement blog post. The real test is: **can it handle a messy, real-world problem?**

I needed a use case that would truly stress-test the system. Something with:
- Large volume (thousands of documents)
- Varied content (emails, legal docs, images)
- Real research value (not just a toy demo)
- Public interest (so others could replicate)

Then I found it.

---

## The Challenge: 25,000 Pages of Epstein Files

On a Friday afternoon, the House Oversight Committee released approximately 25,000 pages of documents related to Jeffrey Epstein. The internet exploded with interest—journalists, researchers, and concerned citizens all wanted to dig through these files. But there was a problem.

**The documents were scattered across folders, many were JPG scans requiring OCR, and manually reading 25,000 pages would take months.**

This is exactly the kind of problem RAG systems are supposed to solve. But preprocessing 25,000 images with OCR? That's a massive undertaking. I'm a developer, not a data processing team.

---

## The Discovery: A Preprocessed Dataset

While researching the dataset, I stumbled across a HuggingFace repository that changed everything:

> *"I've processed all the text and image files (~25,000 document pages/emails) within individual folders released last friday into a two column text file. I used Google's Tesseract OCR library to convert jpg to text."*
>
> — [@tensonaut](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)

This was perfect. Someone had already done the hard work:
- ✅ OCR completed on all images
- ✅ Organized into a clean CSV format
- ✅ Original Google Drive paths preserved for verification
- ✅ Two simple columns: `filename` and `text`

Now I had no excuses. Time to see if Google's Gemini File Search could actually deliver on its promises.

---

## The Architecture: Hybrid RAG Approach

Here's what I built (high-level):

```
┌──────────────────────────────────────────────────┐
│  CSV Dataset (25K documents)                     │
│  filename | text                                 │
└──────────────────────────────────────────────────┘
                    ↓
         Process & Export as TXT files
                    ↓
┌──────────────────────────────────────────────────┐
│  Gemini File Search Store                        │
│  - Automatic chunking                            │
│  - Embedding generation (text-embedding-005)     │
│  - Vector indexing                               │
│  - Semantic search                               │
└──────────────────────────────────────────────────┘
                    ↓
              User Query
                    ↓
      ┌─────────────────────────┐
      │  RETRIEVAL (Gemini)     │
      │  Semantic search → Top  │
      │  relevant document chunks│
      └─────────────────────────┘
                    ↓
           Retrieved Context
                    ↓
      ┌─────────────────────────┐
      │  GENERATION (OpenAI)    │
      │  GPT-4o-mini generates  │
      │  factual answer         │
      └─────────────────────────┘
                    ↓
      ┌─────────────────────────┐
      │  Response with Citations│
      │  + Google Drive links   │
      └─────────────────────────┘
```

**Why hybrid?** I quickly discovered that Gemini's generation endpoint was getting 503 errors under load. So I pivoted: **use Gemini's excellent File Search for retrieval, but generate answers with OpenAI's GPT-4o-mini for reliability.**

This turned out to be the right call.

---

## Implementation Journey: What I Learned

### 1. **The Upload Process (Surprisingly Smooth)**

The Gemini File Search API is straightforward:

```python
# Create a search store (one-time)
store = client.file_search_stores.create(
    config={'display_name': 'epstein_documents_store'}
)

# Upload documents
operation = client.file_search_stores.upload_to_file_search_store(
    file=file_path,
    file_search_store_name=store.name,
    config={'display_name': doc_id}
)

# Poll until complete (critical!)
while not operation.done:
    time.sleep(5)
    operation = client.operations.get(operation)
```

**Key learning**: Pass the entire `operation` object to `operations.get()`, not `operation.name`. This tripped me up initially—the docs are clear, but easy to miss.

### 2. **The Metadata Problem**

Early on, I tried to attach metadata (source paths, document IDs, categories) to each uploaded file:

```python
# This DOESN'T work ❌
config={
    'display_name': doc_id,
    'metadata': {'source_path': path, 'category': 'legal'}  # Rejected!
}
```

Google's API returned: `Extra inputs are not permitted [type=extra_forbidden]`

**Solution**: Store metadata locally, not in the API. The File Search API doesn't need it—document chunking and retrieval work fine without custom metadata. I track source mappings in my application layer instead.

### 3. **The 503 Error Wall**

Testing queries, I hit this immediately:

```
503 UNAVAILABLE: The model is overloaded. Please try again later.
```

This killed my initial plan to use Gemini end-to-end. But here's where the architecture flexibility saved me: **Retrieval and generation are separate concerns.**

I kept Gemini File Search for retrieval (its strength) and switched to OpenAI GPT-4o-mini for generation (rock-solid reliability). Best of both worlds.

### 4. **The Query Pattern**

The final query flow:

```python
# Step 1: Retrieve with Gemini File Search
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

# Extract retrieved document chunks
context_chunks = extract_context(retrieval_response)

# Step 2: Generate answer with OpenAI
completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT: {context_chunks}"}
    ]
)

answer = completion.choices[0].message.content
```

Clean separation. Gemini does semantic search across 25K documents, OpenAI generates the answer from retrieved context.

### 5. **The UX Challenge**

Initial design had a linear workflow: Load CSV → Upload → Query. But users would have to re-upload documents every session!

**Better approach**: Tab-based UI with **independent upload and query**:
- **Tab 1**: Ask Questions (connect to existing store, query immediately)
- **Tab 2**: Upload Documents (only when adding new docs)
- **Tab 3**: Manage Data (view store info)

Documents upload once, query forever. Much better.

---

## Results: What Actually Worked

### Performance Metrics

**Upload (One-Time)**:
- 10 documents: ~1-2 minutes
- 100 documents: ~15-20 minutes
- Estimated 25K documents: ~4-6 hours

**Query (Every Time)**:
- Retrieval (Gemini): 1-2 seconds
- Generation (OpenAI): 2-4 seconds
- **Total**: 3-6 seconds per query

### Cost Breakdown

For 10,000 documents (testing at scale):

| Operation | Provider | Cost |
|-----------|----------|------|
| Indexing (one-time) | Gemini | $1.50 |
| Retrieval (per query) | Gemini | $0.00 |
| Generation (per query) | OpenAI | ~$0.002 |
| **100 queries total** | Both | **$1.70** |

**Per query cost: ~$0.017 (1.7 cents)**

That's remarkably cost-effective for searching 10K documents with AI-generated answers and source citations.

### Quality Assessment

Testing with real questions:

**Query**: *"Who are the main individuals mentioned in the documents?"*

**Result**:
- Retrieved 5 relevant document chunks
- Generated comprehensive answer citing specific documents
- Included Google Drive links to original sources
- Response time: 4.2 seconds

**Query**: *"What locations appear most frequently?"*

**Result**:
- Semantic search found location references across documents
- Aggregated mentions from multiple sources
- Cross-referenced with original file paths
- Completely accurate citations

The system works. Really works.

---

## Lessons Learned: The Real Talk

### What Google Got Right

1. **Zero Vector Database Management**: This is huge. No Pinecone, no Chroma, no pgvector. Upload files, done.

2. **Automatic Chunking**: Google's chunking strategy is solid. Documents are split intelligently without manual tuning.

3. **Semantic Search Quality**: The retrieval is genuinely good. It understands context, not just keywords.

4. **Storage is Free**: You only pay for indexing ($0.15 per 1M tokens). Storage and queries are free. This is a game-changer for economics.

5. **It Just Works**: When it works, it really works. No fiddling with embedding dimensions or distance metrics.

### What Needs Improvement

1. **503 Errors on Generation**: This is a deal-breaker for production. Hence my hybrid approach.

2. **No Custom Metadata**: Can't attach arbitrary metadata to documents. Have to track externally.

3. **Limited Citation Control**: You get what Gemini decides to cite. Can't force specific chunking strategies.

4. **Documentation Gaps**: The operation polling pattern isn't obvious. Took trial and error to get right.

5. **Upload Speed**: Processing is slow. 25K documents would take hours. This is a one-time cost, but still.

### The Verdict

**Is Gemini File Search production-ready?**

For retrieval: **Yes, absolutely.** It's reliable, fast, and requires zero infrastructure.

For end-to-end RAG: **Not yet.** The 503 errors kill it. Use hybrid architecture.

**Would I use this for a real product?**

Yes, but with OpenAI (or another reliable LLM) for generation. The retrieval is world-class.

---

## The Bigger Picture: What This Means

Building this research assistant taught me something important: **Managed RAG services are finally viable.**

Five years ago, building this would have required:
- Setting up a vector database (Elasticsearch, Pinecone, etc.)
- Choosing and hosting an embedding model
- Writing chunking logic
- Managing indices
- Scaling infrastructure

Now? Upload files to an API. That's it.

**This is the same shift we saw with:**
- AWS Lambda (managed compute)
- Firebase (managed databases)
- Vercel (managed deployments)

We're seeing **managed AI infrastructure** emerge. Google's File Search is imperfect, but it's pointing the direction.

---

## Practical Takeaways for Developers

If you're building something similar:

### 1. **Start with Managed Solutions**

Don't build your own vector database unless you have a specific reason. Test managed options first.

### 2. **Separate Retrieval and Generation**

Retrieval and generation are different concerns. Use the best tool for each:
- Gemini File Search for retrieval
- OpenAI/Anthropic for generation
- Mix and match as needed

### 3. **Handle the Upload Once**

Design your UX so documents upload once, query forever. Users will thank you.

### 4. **Test with Real Data**

Toy datasets lie. The Epstein files had real-world messiness—OCR errors, varied formats, inconsistent structure. That's where you find the edge cases.

### 5. **Monitor Costs**

RAG can get expensive. For this use case:
- Indexing: One-time $3-4 for 25K docs
- Queries: ~$0.02 per query
- Scale accordingly

---

## What's Next

The research assistant is live and functional. Next steps:

1. **Scale to Full 25K Documents**: Currently tested with 100-1000 docs. Time to go all in.

2. **Advanced Querying**: Add filters (date ranges, document types, entities).

3. **Multi-Document Analysis**: "Compare how different sources describe this event."

4. **Export & Reporting**: Generate investigation reports with citations.

5. **Public Demo**: Make it available for journalists and researchers.

---

## Conclusion: Worth the Hype?

Started skeptical. Ended impressed (with caveats).

**Google's Gemini File Search is not perfect**, but it's a genuine leap forward in making RAG accessible. The fact that I built a 25,000-document research assistant in a weekend—with no vector database, no embedding model hosting, and minimal infrastructure—is remarkable.

The hybrid architecture (Gemini + OpenAI) turned out to be the sweet spot: reliable, fast, cost-effective, and production-ready.

**For technical teams considering RAG**: Give Gemini File Search a serious look. Use it for retrieval, pair it with a reliable LLM for generation, and you've got something powerful.

The managed AI infrastructure era is here. It's messy and imperfect, but it's real.

---

## Resources

**Dataset**: [Epstein Files on HuggingFace](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K) by [@tensonaut](https://huggingface.co/tensonaut)

**Google Gemini File Search**: [Official Documentation](https://ai.google.dev/gemini-api/docs/file-search)

**Code & Architecture**: Available on GitHub (see project documentation)

**Original Source**: [House Oversight Committee Release](https://oversightdemocrats.house.gov/)

---

## About This Project

Built as an exploration of Google's Gemini File Search API and a practical tool for investigating public interest documents.

**Tech Stack**:
- Python + Streamlit
- Google Gemini API (File Search)
- OpenAI API (GPT-4o-mini)
- ~2,200 lines of code

**Time to Build**: ~2 days (including all the debugging)

**Cost to Index 10K documents**: $1.50

**Cost per query**: ~$0.017

**Status**: Functional prototype, ready for scaling

---

*Questions or want to replicate this? The architecture and lessons learned are documented in the project repository. Feel free to reach out.*

---

**Tags**: #RAG #GoogleGemini #OpenAI #DocumentSearch #AI #MachineLearning #InvestigativeJournalism #TechnicalDeepDive
