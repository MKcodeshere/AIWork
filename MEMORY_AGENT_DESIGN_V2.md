# Production-Grade Memory Agent Design for Denodo AI SDK (V2 - Revised)

## Executive Summary

**REVISION:** Based on frontend integration requirements, this design extends the existing `/answerQuestion` endpoint with transparent memory support instead of creating a separate endpoint.

**Key Design Principles:**
- ✅ **Zero frontend breaking changes** - All memory parameters are optional
- ✅ **Backward compatible** - Existing calls work unchanged
- ✅ **Gradual adoption** - Frontend passes `user_id` when ready for memory features
- ✅ **Simple user identification** - Accept `user_id` as request parameter (not extracted from auth)
- ✅ **Feature flag controlled** - Can disable globally via configuration

---

## 1. Design Comparison: V1 vs V2

| Aspect | V1 (Separate Endpoint) | V2 (Transparent Memory) ✅ |
|--------|------------------------|---------------------------|
| **Endpoint** | New `/memoryChat` | Extend `/answerQuestion` |
| **Frontend Changes** | Required (call different endpoint) | Optional (just add params) |
| **User ID** | Extract from auth token | Accept as parameter |
| **Backward Compatibility** | No (new endpoint) | Yes (optional params) |
| **Deployment Complexity** | Higher (route changes) | Lower (same endpoint) |
| **A/B Testing** | Harder (route-based) | Easier (param-based) |

**Winner:** V2 - Transparent memory with extended endpoint

---

## 2. Revised Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Frontend/Chatbot (No Changes!)                   │
│  Calls: /answerQuestion                                             │
│  Old:   { question, llm_model, ... }                                │
│  New:   { question, llm_model, user_id, session_id, ... }  ← OPTIONAL│
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              EXTENDED: /answerQuestion Endpoint                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ if user_id provided AND MEMORY_ENABLED:                      │  │
│  │   1. Get/create conversation session                         │  │
│  │   2. Load conversation history from PostgreSQL               │  │
│  │   3. Semantic search on user's memory (PGVector)             │  │
│  │   4. Build augmented context (history + similar queries)     │  │
│  │   5. Process with EXISTING answerQuestion logic              │  │
│  │   6. Store Q&A in memory (DB + vector store)                 │  │
│  │   7. Return response + conversation_metadata                 │  │
│  │ else:                                                         │  │
│  │   → ORIGINAL flow (no memory, existing behavior)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ PostgreSQL   │  │ PGVector     │  │ Existing     │
│ Conversations│  │ Semantic     │  │ VQL Logic    │
│ & Messages   │  │ Memory       │  │ Unchanged    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 2.2 Detailed Sequence Diagram

```
User     Chatbot    /answerQuestion    MemoryManager    PGVector    PostgreSQL    VQLGenerator    DataCatalog
 │          │             │                  │              │              │              │              │
 │ "What about 2023?" (with user_id="john@example.com", session_id="abc123")               │              │
 │──────────────────────> │                  │              │              │              │              │
 │          │             │                  │              │              │              │              │
 │          │       ┌─────┴────────┐         │              │              │              │              │
 │          │       │ Check params │         │              │              │              │              │
 │          │       │ user_id? ✓   │         │              │              │              │              │
 │          │       │ MEMORY_ON? ✓ │         │              │              │              │              │
 │          │       └─────┬────────┘         │              │              │              │              │
 │          │             │                  │              │              │              │              │
 │          │             │──1. get_or_create_conversation(user_id, session_id)──────────>│              │
 │          │             │                  │              │              │              │              │
 │          │             │<─────Returns conversation{id, created_at, expires_at}─────────│              │
 │          │             │                  │              │              │              │              │
 │          │             │──2. get_conversation_history(conversation_id, last_10)────────>│              │
 │          │             │                  │              │              │              │              │
 │          │             │<─Returns [msg1: "sales 2024", msg2: "by region", ...]─────────│              │
 │          │             │                  │              │              │              │              │
 │          │             │──3. semantic_search_memory(user_id, "What about 2023?")───────>│              │
 │          │             │                  │              │              │              │              │
 │          │             │<─Returns similar queries: [{question: "sales by region 2024",  │              │
 │          │             │  sql: "SELECT...", tables: [sales, regions]}]──────────────────│              │
 │          │             │                  │              │              │              │              │
 │          │       ┌─────┴────────┐         │              │              │              │              │
 │          │       │4. Build      │         │              │              │              │              │
 │          │       │   augmented  │         │              │              │              │              │
 │          │       │   context:   │         │              │              │              │              │
 │          │       │               │         │              │              │              │              │
 │          │       │ "CONTEXT:    │         │              │              │              │              │
 │          │       │  User asked: │         │              │              │              │              │
 │          │       │  'sales by   │         │              │              │              │              │
 │          │       │  region 2024'│         │              │              │              │              │
 │          │       │  Tables: sales│        │              │              │              │              │
 │          │       │  regions     │         │              │              │              │              │
 │          │       │               │         │              │              │              │              │
 │          │       │ NOW: What    │         │              │              │              │              │
 │          │       │ about 2023?" │         │              │              │              │              │
 │          │       └─────┬────────┘         │              │              │              │              │
 │          │             │                  │              │              │              │              │
 │          │             │──5. EXISTING answerQuestion logic (augmented_question)─────────>│              │
 │          │             │                  │              │              │              │              │
 │          │             │                  │              │              │              │──6. Vector───>│
 │          │             │                  │              │              │              │   search     │
 │          │             │                  │              │              │              │   tables     │
 │          │             │                  │              │              │              │<─────────────│
 │          │             │                  │              │              │              │              │
 │          │             │                  │              │              │              │──7. Gen VQL──>│
 │          │             │                  │              │              │              │   for 2023   │
 │          │             │                  │              │              │              │<─────────────│
 │          │             │                  │              │              │              │              │
 │          │             │<─8. Returns {answer, sql, tables_used, execution_result}───────│              │
 │          │             │                  │              │              │              │              │
 │          │       ┌─────┴────────┐         │              │              │              │              │
 │          │       │9. Store      │         │              │              │              │              │
 │          │       │   interaction│         │              │              │              │              │
 │          │       └─────┬────────┘         │              │              │              │              │
 │          │             │                  │              │              │              │              │
 │          │             │──10. store_interaction(conv_id, user_id, question, result)─────>│              │
 │          │             │                  │              │              │              │              │
 │          │             │                  │──11. Insert message to DB──────────────────>│              │
 │          │             │                  │                                             │              │
 │          │             │                  │──12. Embed & store vector──>│              │              │
 │          │             │                  │                             │              │              │
 │          │<─13. Returns response + conversation_metadata{session_id, message_count}─────│              │
 │          │             │                  │              │              │              │              │
 │<──────────────────────────────────────────│              │              │              │              │
 │          │             │                  │              │              │              │              │


SCENARIO 2: No user_id provided (Original Behavior)
─────────────────────────────────────────────────────

User     Chatbot    /answerQuestion    VQLGenerator    DataCatalog
 │          │             │                  │              │
 │ "Show me sales 2024" (NO user_id)         │              │
 │──────────────────────> │                  │              │
 │          │             │                  │              │
 │          │       ┌─────┴────────┐         │              │
 │          │       │ Check params │         │              │
 │          │       │ user_id? ✗   │         │              │
 │          │       │ → SKIP MEMORY│         │              │
 │          │       └─────┬────────┘         │              │
 │          │             │                  │              │
 │          │             │──EXISTING answerQuestion logic───>│
 │          │             │                  │              │
 │          │             │                  │──Vector──────>│
 │          │             │                  │  search      │
 │          │             │                  │  & execute   │
 │          │             │                  │<─────────────│
 │          │             │                  │              │
 │          │             │<─Returns standard response───────│
 │          │             │  (NO conversation_metadata)      │
 │          │<────────────│                  │              │
 │          │             │                  │              │
```

---

## 3. API Specification: Extended answerQuestion Endpoint

### 3.1 Request Schema (BACKWARD COMPATIBLE)

```python
# File: /tmp/denodo-ai-sdk/api/endpoints/answerQuestion.py

class answerQuestionRequest(BaseModel):
    # EXISTING PARAMETERS (unchanged)
    question: str
    plot: bool = False
    plot_details: str = ''
    embeddings_provider: str
    embeddings_model: str
    vector_store_provider: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    vdp_database_names: str
    vdp_tag_names: str
    use_views: str
    expand_set_views: bool
    custom_instructions: str
    markdown_response: bool
    vector_search_k: int = 5
    vector_search_sample_data_k: int = 3
    mode: Literal["default", "data", "metadata"]
    disclaimer: bool
    verbose: bool
    vql_execute_rows_limit: int
    llm_response_rows_limit: int

    # ═══════════════════════════════════════════════════
    # NEW: MEMORY PARAMETERS (all optional)
    # ═══════════════════════════════════════════════════

    user_id: Optional[str] = None
    """
    User identifier for conversation memory.
    - If provided + MEMORY_ENABLED: activate memory features
    - If None: use original behavior (no memory)
    - Examples: "john@example.com", "user123", "dept_finance"
    """

    session_id: Optional[str] = None
    """
    Conversation session identifier.
    - If provided: resume existing conversation
    - If None + user_id provided: create new conversation
    - Use this to maintain context across multiple questions
    - Frontend should persist this ID across user interactions
    """

    use_memory: bool = True
    """
    Enable/disable memory for this specific query.
    - Default: True (use memory if user_id provided)
    - Set to False to force non-memory behavior even with user_id
    - Useful for testing or specific use cases
    """

    max_history_messages: Optional[int] = None
    """
    Override number of recent messages to include in context.
    - Default: Uses MEMORY_MAX_HISTORY_MESSAGES from config (10)
    - Range: 1-50
    """

    semantic_search_k: Optional[int] = None
    """
    Override number of similar past queries to retrieve.
    - Default: Uses MEMORY_SEMANTIC_SEARCH_K from config (3)
    - Range: 0-10 (0 = disable semantic search)
    """
```

### 3.2 Response Schema (BACKWARD COMPATIBLE)

```python
class answerQuestionResponse(BaseModel):
    # EXISTING FIELDS (unchanged)
    answer: str
    sql_query: str
    query_explanation: str
    tokens: Dict
    execution_result: Dict
    related_questions: List[str]
    tables_used: List[str]
    raw_graph: str
    sql_execution_time: float
    vector_store_search_time: float
    llm_time: float
    total_execution_time: float

    # ═══════════════════════════════════════════════════
    # NEW: MEMORY METADATA (null if memory not used)
    # ═══════════════════════════════════════════════════

    conversation_metadata: Optional[Dict[str, Any]] = None
    """
    Present only when memory is used.
    Schema:
    {
        "conversation_id": "uuid-string",
        "session_id": "abc123",  # Same as request or newly created
        "session_name": "Show me total sales...",  # First question
        "message_count": 5,  # Total messages in this conversation
        "created_at": "2025-11-13T10:00:00Z",
        "updated_at": "2025-11-13T10:30:00Z",
        "expires_at": "2025-11-20T10:00:00Z",  # created_at + RETENTION_DAYS
        "is_summarized": false
    }
    """

    memory_context_used: Optional[Dict[str, Any]] = None
    """
    Present only when memory is used.
    Schema:
    {
        "history_messages_count": 4,  # How many past messages included
        "similar_queries_count": 2,   # How many semantic matches found
        "context_tokens": 856,        # Tokens used for context augmentation
        "context_truncated": false,   # True if context exceeded max tokens
        "memory_enabled": true        # Confirms memory was active
    }
    """

    memory_retrieval_time: Optional[float] = None
    """
    Time in seconds to retrieve and process memory.
    Present only when memory is used.
    Includes: DB query + vector search + context building time
    """
```

### 3.3 Example Usage Scenarios

#### Scenario 1: First Question (Create New Conversation)

**Request:**
```json
POST /answerQuestion
{
    "question": "Show me total sales by region for 2024",
    "user_id": "john@example.com",
    "session_id": null,
    "vdp_database_names": "banking_db",
    "llm_provider": "OpenAI",
    "llm_model": "gpt-4o",
    "llm_temperature": 0.0,
    "llm_max_tokens": 4096,
    "embeddings_provider": "OpenAI",
    "embeddings_model": "text-embedding-3-large",
    "vector_store_provider": "pgvector",
    "use_memory": true
    // ... other existing params
}
```

**Response:**
```json
{
    "answer": "The total sales by region for 2024 are:\n- North: $2.5M\n- South: $1.8M\n- East: $2.1M\n- West: $1.6M",
    "sql_query": "SELECT r.region_name, SUM(s.amount) AS total_sales FROM sales s JOIN regions r ON s.region_id = r.region_id WHERE YEAR(s.sale_date) = 2024 GROUP BY r.region_name",
    "query_explanation": "This query joins the sales and regions tables...",
    "tables_used": ["sales", "regions"],
    "execution_result": { /* query results */ },
    "tokens": { "prompt_tokens": 1245, "completion_tokens": 456 },
    "sql_execution_time": 0.234,
    "vector_store_search_time": 0.089,
    "llm_time": 2.145,
    "total_execution_time": 2.468,

    // NEW: Memory metadata
    "conversation_metadata": {
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "session_id": "550e8400-e29b-41d4-a716-446655440000",  // Auto-generated
        "session_name": "Show me total sales by region for 2024",
        "message_count": 1,
        "created_at": "2025-11-13T10:00:00Z",
        "updated_at": "2025-11-13T10:00:00Z",
        "expires_at": "2025-11-20T10:00:00Z",
        "is_summarized": false
    },
    "memory_context_used": {
        "history_messages_count": 0,  // First message
        "similar_queries_count": 0,   // No prior queries
        "context_tokens": 0,
        "context_truncated": false,
        "memory_enabled": true
    },
    "memory_retrieval_time": 0.012
}
```

#### Scenario 2: Follow-up Question (Resume Conversation)

**Request:**
```json
POST /answerQuestion
{
    "question": "What about 2023?",
    "user_id": "john@example.com",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",  // From previous response
    "vdp_database_names": "banking_db",
    // ... same params as before
}
```

**Response:**
```json
{
    "answer": "The total sales by region for 2023 are:\n- North: $2.2M\n- South: $1.5M\n- East: $1.9M\n- West: $1.4M",
    "sql_query": "SELECT r.region_name, SUM(s.amount) AS total_sales FROM sales s JOIN regions r ON s.region_id = r.region_id WHERE YEAR(s.sale_date) = 2023 GROUP BY r.region_name",
    "tables_used": ["sales", "regions"],
    // ... other fields

    "conversation_metadata": {
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "session_name": "Show me total sales by region for 2024",
        "message_count": 2,  // Incremented
        "updated_at": "2025-11-13T10:05:00Z",  // Updated
        "expires_at": "2025-11-20T10:05:00Z"   // Extended based on last interaction
    },
    "memory_context_used": {
        "history_messages_count": 1,  // Previous question included
        "similar_queries_count": 1,   // Found similar query
        "context_tokens": 245,        // Context added
        "context_truncated": false,
        "memory_enabled": true
    },
    "memory_retrieval_time": 0.045
}
```

#### Scenario 3: No Memory (Original Behavior)

**Request:**
```json
POST /answerQuestion
{
    "question": "Show me customer data",
    // NO user_id parameter
    "vdp_database_names": "banking_db",
    // ... other params
}
```

**Response:**
```json
{
    "answer": "Here are the customers...",
    "sql_query": "SELECT * FROM customers LIMIT 100",
    // ... standard fields

    // NO memory fields
    "conversation_metadata": null,
    "memory_context_used": null,
    "memory_retrieval_time": null
}
```

---

## 4. Implementation: Code Changes

### 4.1 Modified File: `/tmp/denodo-ai-sdk/api/endpoints/answerQuestion.py`

**Current Structure (Lines 147-234):**
```python
@router.post("/answerQuestion", response_model=answerQuestionResponse)
async def answer_question(request: answerQuestionRequest, auth=Depends(authenticate)):
    # Initialize resources
    llm = state_manager.get_llm(...)
    vector_store = state_manager.get_vector_store(...)

    # Get relevant tables
    relevant_tables = get_relevant_tables(...)

    # Process based on category
    if category == "SQL":
        result = process_sql_category(...)
    else:
        result = process_metadata_category(...)

    return result
```

**NEW Structure (With Memory Integration):**
```python
@router.post("/answerQuestion", response_model=answerQuestionResponse)
async def answer_question(request: answerQuestionRequest, auth=Depends(authenticate)):

    # ═══════════════════════════════════════════════════
    # NEW: Memory Pre-Processing
    # ═══════════════════════════════════════════════════
    memory_enabled = (
        request.user_id is not None and
        request.use_memory and
        os.getenv("MEMORY_ENABLED", "0") == "1"
    )

    conversation_metadata = None
    memory_context_used = None
    memory_retrieval_start = time.time()
    augmented_question = request.question  # Default: no augmentation

    if memory_enabled:
        try:
            memory_manager = state_manager.get_memory_manager()

            # 1. Get or create conversation
            conversation = memory_manager.get_or_create_conversation(
                user_id=request.user_id,
                session_id=request.session_id,
                vdp_database_names=request.vdp_database_names.split(',') if request.vdp_database_names else []
            )

            # 2. Load conversation history
            max_history = request.max_history_messages or int(os.getenv("MEMORY_MAX_HISTORY_MESSAGES", "10"))
            history = memory_manager.get_conversation_history(
                conversation_id=conversation.id,
                max_messages=max_history
            )

            # 3. Semantic search for similar queries
            search_k = request.semantic_search_k or int(os.getenv("MEMORY_SEMANTIC_SEARCH_K", "3"))
            similar_memories = memory_manager.semantic_search_memory(
                user_id=request.user_id,
                query=request.question,
                k=search_k,
                conversation_id=conversation.id  # Optional: scope to current conversation
            )

            # 4. Build augmented context
            if history or similar_memories:
                augmented_question = memory_manager.build_augmented_context(
                    current_question=request.question,
                    history=history,
                    similar_memories=similar_memories,
                    max_tokens=int(os.getenv("MEMORY_MAX_CONTEXT_TOKENS", "4000"))
                )

            # Track memory usage
            memory_context_used = {
                "history_messages_count": len(history),
                "similar_queries_count": len(similar_memories),
                "context_tokens": count_tokens(augmented_question) - count_tokens(request.question),
                "context_truncated": False,  # TODO: Implement truncation logic
                "memory_enabled": True
            }

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            # Fallback: Continue without memory
            memory_enabled = False
            augmented_question = request.question

    memory_retrieval_time = time.time() - memory_retrieval_start if memory_enabled else None

    # ═══════════════════════════════════════════════════
    # EXISTING: answerQuestion Logic (uses augmented_question)
    # ═══════════════════════════════════════════════════

    # Initialize resources
    llm = state_manager.get_llm(...)
    vector_store = state_manager.get_vector_store(...)

    # Get relevant tables (using augmented question)
    relevant_tables = get_relevant_tables(
        question=augmented_question,  # <-- Use augmented if memory enabled
        vector_store=vector_store,
        ...
    )

    # Categorize and process
    category = categorize_question(augmented_question, ...)

    if category == "SQL":
        result = process_sql_category(
            question=augmented_question,  # <-- Augmented question
            original_question=request.question,  # Keep original for logging
            relevant_tables=relevant_tables,
            llm=llm,
            auth=auth,
            ...
        )
    else:
        result = process_metadata_category(
            question=augmented_question,
            ...
        )

    # ═══════════════════════════════════════════════════
    # NEW: Memory Post-Processing (Store Interaction)
    # ═══════════════════════════════════════════════════

    if memory_enabled:
        try:
            memory_manager.store_interaction(
                conversation_id=conversation.id,
                user_id=request.user_id,
                question=request.question,  # Store original question
                answer_data={
                    "answer": result["answer"],
                    "sql_query": result.get("sql_query"),
                    "query_explanation": result.get("query_explanation"),
                    "tables_used": result.get("tables_used", []),
                    "execution_result": result.get("execution_result"),
                    "tokens": result.get("tokens"),
                    "execution_time_ms": result.get("sql_execution_time", 0) * 1000
                }
            )

            # Build conversation metadata for response
            conversation_metadata = {
                "conversation_id": str(conversation.id),
                "session_id": request.session_id or str(conversation.id),
                "session_name": conversation.session_name,
                "message_count": conversation.message_count + 1,  # Will be incremented by store_interaction
                "created_at": conversation.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "expires_at": conversation.expires_at.isoformat(),
                "is_summarized": conversation.is_summarized
            }

        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            # Don't fail the request, just log the error

    # ═══════════════════════════════════════════════════
    # Return Response (with memory metadata if applicable)
    # ═══════════════════════════════════════════════════

    return answerQuestionResponse(
        **result,  # All existing fields
        conversation_metadata=conversation_metadata,
        memory_context_used=memory_context_used,
        memory_retrieval_time=memory_retrieval_time
    )
```

### 4.2 New File: `/tmp/denodo-ai-sdk/api/utils/state_manager.py` (Modifications)

**Add to existing caches:**
```python
# Existing caches
_llm_cache = {}
_embedding_model_cache = {}
_vector_store_cache = {}

# NEW: Memory manager cache (singleton per configuration)
_memory_manager_cache = None

def get_memory_manager() -> MemoryManager:
    """
    Get or create the singleton MemoryManager instance.
    """
    global _memory_manager_cache

    if _memory_manager_cache is None:
        from api.utils.memory_manager import MemoryManager
        from api.utils.db.connection import get_db_engine
        from utils.uniformEmbedding import get_embedding_model
        from utils.uniformVectorStore import get_vector_store

        # Initialize database connection
        db_engine = get_db_engine()

        # Initialize embeddings for memory vector store
        memory_embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "OpenAI")
        memory_embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
        embeddings = get_embedding_model(memory_embeddings_provider, memory_embeddings_model)

        # Initialize memory vector store
        memory_vector_store_provider = os.getenv("MEMORY_VECTOR_STORE", "pgvector")
        memory_vector_store = get_vector_store(
            provider=memory_vector_store_provider,
            embeddings_provider=memory_embeddings_provider,
            embeddings_model=memory_embeddings_model,
            index_name=os.getenv("MEMORY_VECTOR_COLLECTION_NAME", "ai_sdk_conversation_memory"),
            rate_limit_rpm=os.getenv("RATE_LIMIT_RPM")
        )

        _memory_manager_cache = MemoryManager(
            db_engine=db_engine,
            vector_store=memory_vector_store,
            embeddings_model=embeddings,
            config={
                "retention_days": int(os.getenv("MEMORY_RETENTION_DAYS", "7")),
                "max_history_messages": int(os.getenv("MEMORY_MAX_HISTORY_MESSAGES", "10")),
                "semantic_search_k": int(os.getenv("MEMORY_SEMANTIC_SEARCH_K", "3")),
                "max_context_tokens": int(os.getenv("MEMORY_MAX_CONTEXT_TOKENS", "4000")),
                "summarization_enabled": os.getenv("MEMORY_SUMMARIZATION_ENABLED", "1") == "1",
                "summarization_threshold": int(os.getenv("MEMORY_SUMMARIZATION_THRESHOLD", "15"))
            }
        )

    return _memory_manager_cache

def initialize_default_resources():
    """
    Called at application startup to pre-warm caches.
    """
    # Existing initialization
    get_llm(...)
    get_vector_store(...)

    # NEW: Initialize memory manager if enabled
    if os.getenv("MEMORY_ENABLED", "0") == "1":
        try:
            get_memory_manager()
            logger.info("Memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
```

---

## 5. Database Schema (Unchanged from V1)

**Same as V1 design** - See MEMORY_AGENT_DESIGN.md Section 2.3.A

Key tables:
- `users` (user_id, created_at, last_active_at)
- `conversations` (conversation_id, user_id, session_name, expires_at, message_count)
- `messages` (message_id, conversation_id, content, sql_query, tables_used, execution_result)
- `memory_embeddings` (embedding_id, message_id, user_id, embedding, search_text)

---

## 6. Configuration Updates

### 6.1 File: `/tmp/denodo-ai-sdk/api/utils/sdk_config_example.env`

**Add new section after existing sections:**

```bash
##################################################################################
# Section 8: Conversation Memory (Short-term Context)
##################################################################################

# Enable persistent conversation memory for follow-up questions
# When enabled, answerQuestion endpoint will check for user_id parameter
# and use conversation history to enhance context understanding
MEMORY_ENABLED = 1

# ═══════════════════════════════════════════════════
# Database Configuration
# ═══════════════════════════════════════════════════

# PostgreSQL connection for storing conversation history
# Required format: postgresql+psycopg://username:password@host:port/database
MEMORY_DB_CONNECTION_STRING = postgresql+psycopg://denodo_ai:password@localhost:5432/denodo_ai_sdk_memory

# Connection pool settings (adjust based on expected concurrent users)
MEMORY_DB_POOL_SIZE = 10
MEMORY_DB_MAX_OVERFLOW = 20
MEMORY_DB_POOL_TIMEOUT = 30
MEMORY_DB_ECHO = 0  # Set to 1 for SQL query logging (debug mode)

# ═══════════════════════════════════════════════════
# Vector Store for Semantic Memory
# ═══════════════════════════════════════════════════

# Vector store provider for conversation embeddings (separate from schema metadata)
# Options: pgvector (recommended), chroma, opensearch
MEMORY_VECTOR_STORE = pgvector

# Collection/index name for conversation memory
MEMORY_VECTOR_COLLECTION_NAME = ai_sdk_conversation_memory

# If using PGVector, uses same connection as MEMORY_DB_CONNECTION_STRING
# If using Chroma or OpenSearch, configure below:
# MEMORY_CHROMA_PERSIST_DIRECTORY = ./chroma_memory
# MEMORY_OPENSEARCH_URL = http://localhost:9200

# ═══════════════════════════════════════════════════
# Memory Retention & Cleanup
# ═══════════════════════════════════════════════════

# Conversation retention period (short-term memory)
# After this period, conversations are automatically deleted
# Recommended: 7-10 days for short-term context
MEMORY_RETENTION_DAYS = 7

# How often to run cleanup job (in hours)
# Background task that deletes expired conversations
MEMORY_CLEANUP_INTERVAL_HOURS = 24

# Cleanup batch size (number of conversations to delete per batch)
MEMORY_CLEANUP_BATCH_SIZE = 100

# ═══════════════════════════════════════════════════
# Memory Retrieval Settings
# ═══════════════════════════════════════════════════

# Maximum number of recent messages to include in context
# Higher = more context, but more tokens consumed
# Range: 1-50, Recommended: 10
MEMORY_MAX_HISTORY_MESSAGES = 10

# Number of similar past queries to retrieve via semantic search
# Higher = better context matching, but more noise
# Range: 0-10 (0 = disable semantic search), Recommended: 3
MEMORY_SEMANTIC_SEARCH_K = 3

# Maximum tokens for augmented context (includes history + similar queries)
# Should be less than LLM_MAX_TOKENS to leave room for generation
# Recommended: 25-50% of LLM_MAX_TOKENS
MEMORY_MAX_CONTEXT_TOKENS = 4000

# Minimum similarity threshold for semantic search (cosine similarity)
# Range: 0.0-1.0, Recommended: 0.7
# Queries below this threshold won't be included in context
MEMORY_SEMANTIC_SIMILARITY_THRESHOLD = 0.7

# ═══════════════════════════════════════════════════
# Conversation Summarization
# ═══════════════════════════════════════════════════

# Enable automatic summarization of long conversations
# When conversation exceeds threshold, older messages are summarized
MEMORY_SUMMARIZATION_ENABLED = 1

# Number of messages before triggering summarization
# Recommended: 15-20 messages
MEMORY_SUMMARIZATION_THRESHOLD = 15

# LLM model to use for summarization (recommend cheaper/faster model)
MEMORY_SUMMARIZATION_MODEL = gpt-4o-mini
MEMORY_SUMMARIZATION_PROVIDER = OpenAI
MEMORY_SUMMARIZATION_TEMPERATURE = 0.3
MEMORY_SUMMARIZATION_MAX_TOKENS = 500

# ═══════════════════════════════════════════════════
# Privacy & Compliance
# ═══════════════════════════════════════════════════

# Enable user data export (GDPR compliance)
MEMORY_ALLOW_DATA_EXPORT = 1

# Enable user data deletion / right to be forgotten (GDPR compliance)
MEMORY_ALLOW_DATA_DELETION = 1

# Log all memory operations for audit trail
MEMORY_LOG_OPERATIONS = 1

# Mask sensitive data in logs (PII detection)
MEMORY_MASK_SENSITIVE_DATA = 1

# ═══════════════════════════════════════════════════
# Performance & Optimization
# ═══════════════════════════════════════════════════

# Enable caching of recent conversations in Redis (optional)
# Reduces DB load for active users
MEMORY_REDIS_CACHE_ENABLED = 0
MEMORY_REDIS_URL = redis://localhost:6379/0
MEMORY_REDIS_TTL_SECONDS = 3600  # 1 hour

# Batch size for embedding generation (parallel processing)
MEMORY_EMBEDDING_BATCH_SIZE = 10

# Enable async memory storage (non-blocking)
# Stores interactions in background after response sent
MEMORY_ASYNC_STORAGE = 1

# ═══════════════════════════════════════════════════
# Experimental Features
# ═══════════════════════════════════════════════════

# Enable cross-conversation learning (beta)
# Uses other users' anonymized queries for context (privacy-preserving)
MEMORY_ENABLE_CROSS_USER_LEARNING = 0

# Enable automatic question suggestion based on conversation
MEMORY_ENABLE_SUGGESTIONS = 1
MEMORY_SUGGESTION_COUNT = 3
```

---

## 7. Frontend Integration Guide

### 7.1 Minimal Changes (Gradual Adoption)

**Step 1: No changes required initially**
- Existing calls continue to work
- No memory features active
- Zero breaking changes

**Step 2: Add user_id to enable memory**
```javascript
// Before (no memory)
const response = await fetch('/answerQuestion', {
    method: 'POST',
    body: JSON.stringify({
        question: userInput,
        llm_provider: 'OpenAI',
        llm_model: 'gpt-4o',
        // ... other params
    })
});

// After (with memory)
const response = await fetch('/answerQuestion', {
    method: 'POST',
    body: JSON.stringify({
        question: userInput,
        user_id: getCurrentUserId(),  // <-- ADD THIS
        session_id: getSessionId(),   // <-- ADD THIS (optional, for resume)
        llm_provider: 'OpenAI',
        llm_model: 'gpt-4o',
        // ... other params
    })
});
```

**Step 3: Persist session_id across questions**
```javascript
// Chatbot state management
let currentSessionId = null;

async function askQuestion(userInput) {
    const response = await fetch('/answerQuestion', {
        method: 'POST',
        body: JSON.stringify({
            question: userInput,
            user_id: getCurrentUserId(),
            session_id: currentSessionId,  // null for first question
            // ... other params
        })
    });

    const data = await response.json();

    // Save session_id for follow-up questions
    if (data.conversation_metadata) {
        currentSessionId = data.conversation_metadata.session_id;

        // Optional: Display conversation metadata in UI
        console.log(`Conversation: ${data.conversation_metadata.message_count} messages`);
        console.log(`Expires: ${data.conversation_metadata.expires_at}`);
    }

    return data;
}

// Reset conversation (new topic)
function startNewConversation() {
    currentSessionId = null;
}
```

### 7.2 Enhanced UI Features (Optional)

**Display conversation context:**
```javascript
if (data.memory_context_used) {
    const contextInfo = `
        Using context from ${data.memory_context_used.history_messages_count} previous messages
        and ${data.memory_context_used.similar_queries_count} similar queries
    `;
    showTooltip(contextInfo);
}
```

**Show "New Conversation" button:**
```html
<button onclick="startNewConversation()">
    Start New Topic
</button>
```

**Session management:**
```javascript
// Store session_id in localStorage for cross-browser-session resume
function saveSession() {
    localStorage.setItem('denodo_ai_session_id', currentSessionId);
}

function loadSession() {
    currentSessionId = localStorage.getItem('denodo_ai_session_id');
}

// Clear old sessions
function clearExpiredSessions() {
    // Check expires_at, clear if past
}
```

---

## 8. Implementation Timeline

**Same 8-day timeline as V1, with adjusted tasks:**

### Phase 1: Database Infrastructure (Day 1-2)
- Set up PostgreSQL database
- Create schema (users, conversations, messages, memory_embeddings)
- Test connection and ORM models

### Phase 2: Memory Manager Core (Day 3-4)
- Implement MemoryManager class
- Build conversation CRUD operations
- Implement context augmentation logic
- Unit tests

### Phase 3: Endpoint Integration (Day 5-6)
- **Modify** `/api/endpoints/answerQuestion.py` (not create new endpoint)
- Add memory parameters to request/response models
- Integrate memory pre/post processing
- Integration tests

### Phase 4: TTL & Maintenance (Day 7)
- Implement cleanup job
- Add conversation summarization
- Performance optimization

### Phase 5: Configuration & Documentation (Day 8)
- Update sdk_config_example.env
- Document frontend integration
- Create migration guide

---

## 9. Migration & Rollout

### Phase 1: Infrastructure (Week 1)
1. Provision PostgreSQL database
2. Run schema creation script
3. Configure sdk_config.env with database connection
4. Test connectivity

### Phase 2: Backend Deployment (Week 2)
1. Deploy updated answerQuestion endpoint
2. Set `MEMORY_ENABLED = 0` initially (disabled)
3. Verify backward compatibility (existing calls work)

### Phase 3: Beta Testing (Week 3)
1. Enable memory: `MEMORY_ENABLED = 1`
2. Test with internal users passing `user_id`
3. Monitor logs for errors
4. Tune parameters (semantic_search_k, max_history_messages)

### Phase 4: Frontend Rollout (Week 4)
1. Update frontend to pass `user_id` parameter
2. Gradual rollout: 10% → 50% → 100% of users
3. Feature flag in frontend for A/B testing
4. Collect user feedback

### Phase 5: Optimization (Week 5+)
1. Tune prompt engineering for context augmentation
2. Implement conversation summarization
3. Add Redis caching for hot conversations
4. Monitor performance metrics

---

## 10. Testing Strategy

### 10.1 Backward Compatibility Tests

**Test 1: Existing call without user_id**
```python
def test_backward_compatibility():
    response = client.post("/answerQuestion", json={
        "question": "Show me sales",
        # NO user_id
        "llm_provider": "OpenAI",
        "llm_model": "gpt-4o",
        # ... other required params
    })

    assert response.status_code == 200
    assert response.json()["answer"] is not None
    assert response.json()["conversation_metadata"] is None  # No memory
    assert response.json()["memory_context_used"] is None
```

### 10.2 Memory Feature Tests

**Test 2: First question with user_id**
```python
def test_first_question_creates_conversation():
    response = client.post("/answerQuestion", json={
        "question": "Show me sales by region for 2024",
        "user_id": "test_user@example.com",
        "session_id": None,
        # ... other params
    })

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_metadata"] is not None
    assert data["conversation_metadata"]["message_count"] == 1
    assert data["conversation_metadata"]["session_id"] is not None

    # Save session_id for next test
    return data["conversation_metadata"]["session_id"]
```

**Test 3: Follow-up question with context**
```python
def test_followup_question_uses_context(session_id):
    response = client.post("/answerQuestion", json={
        "question": "What about 2023?",
        "user_id": "test_user@example.com",
        "session_id": session_id,
        # ... other params
    })

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_metadata"]["message_count"] == 2
    assert data["memory_context_used"]["history_messages_count"] >= 1

    # Verify SQL query uses correct tables from context
    assert "sales" in data["sql_query"].lower()
    assert "region" in data["sql_query"].lower()
    assert "2023" in data["sql_query"]
```

**Test 4: Memory isolation (cross-user)**
```python
def test_memory_isolation():
    # User 1 creates conversation
    response1 = client.post("/answerQuestion", json={
        "question": "Show me confidential data",
        "user_id": "user1@example.com",
        # ...
    })

    # User 2 should NOT see user 1's history
    response2 = client.post("/answerQuestion", json={
        "question": "Show me data",
        "user_id": "user2@example.com",
        # ...
    })

    assert response2.json()["memory_context_used"]["history_messages_count"] == 0
```

### 10.3 Performance Tests

```python
def test_memory_latency_overhead():
    # Measure without memory
    start = time.time()
    response_no_memory = client.post("/answerQuestion", json={
        "question": "Show me sales",
        # NO user_id
    })
    time_no_memory = time.time() - start

    # Measure with memory
    start = time.time()
    response_with_memory = client.post("/answerQuestion", json={
        "question": "Show me sales",
        "user_id": "perf_test_user",
    })
    time_with_memory = time.time() - start

    # Memory overhead should be < 100ms
    memory_overhead = time_with_memory - time_no_memory
    assert memory_overhead < 0.1, f"Memory overhead too high: {memory_overhead}s"

    # Check reported retrieval time
    assert response_with_memory.json()["memory_retrieval_time"] < 0.1
```

---

## 11. Key Differences from V1

| Aspect | V1 | V2 ✅ |
|--------|-----|------|
| **Endpoint** | New `/memoryChat` | Extended `/answerQuestion` |
| **Frontend Impact** | Must change endpoint URL | Optional parameters |
| **User ID Source** | Extract from auth token | Accept as parameter |
| **Deployment** | New route registration | Modify existing route |
| **Rollback Plan** | Remove new endpoint | Set `MEMORY_ENABLED=0` |
| **A/B Testing** | Route-based | Parameter-based |
| **Code Changes** | New file | Modify existing file |

---

## 12. Decision Summary

### ✅ Finalized Decisions

1. **Architecture:** Extend existing `/answerQuestion` endpoint (not separate endpoint)
2. **User ID:** Accept as request parameter `user_id: Optional[str]`
3. **Backward Compatibility:** All memory parameters optional
4. **Database:** PostgreSQL with PGVector (same as metadata store)
5. **Feature Flag:** `MEMORY_ENABLED` in sdk_config.env
6. **Session Management:** Frontend persists `session_id` across questions
7. **Error Handling:** If memory fails, fallback to non-memory behavior (log error)

### ❓ Open Questions (Need Confirmation)

1. **Default Behavior:** When `user_id` provided, should memory be enabled by default? (Recommend: Yes, with `use_memory=True`)
2. **Session ID:** Auto-generate if not provided? (Recommend: Yes, use conversation_id)
3. **TTL Extension:** Should each new message extend expires_at? (Recommend: Yes, sliding window)
4. **Redis Caching:** Optional performance optimization? (Recommend: Phase 2)
5. **Cross-Database Memory:** Should conversations be scoped per VDP database or global? (Recommend: Global, filter during retrieval)

---

## 13. Success Metrics (Same as V1)

**Before Implementation:**
- Follow-up question success rate: ~30%
- Average questions per session: 1.5
- User satisfaction: 3.2/5

**After Implementation (Target):**
- Follow-up question success rate: >80%
- Average questions per session: 4-5
- User satisfaction: >4.0/5
- Memory hit rate: >70%
- Latency increase: <50ms (p95)

---

## 14. Next Steps

1. ✅ **Approve V2 Design** - Confirm transparent memory approach
2. **Answer Open Questions** - Finalize configuration defaults
3. **Provision PostgreSQL** - Set up database infrastructure
4. **Begin Implementation** - Start with database layer (Phase 1)
5. **Update Configuration** - Add memory settings to sdk_config.env
6. **Modify answerQuestion Endpoint** - Integrate memory logic
7. **Test Thoroughly** - Ensure backward compatibility
8. **Deploy & Monitor** - Gradual rollout with feature flag

---

## Summary

This revised design (V2) provides a **production-grade, backward-compatible, and frontend-friendly** memory agent that:

✅ Extends existing `/answerQuestion` endpoint (no new routes)
✅ Accepts `user_id` as simple request parameter (no auth token parsing)
✅ Requires zero frontend changes initially (gradual adoption)
✅ Maintains full backward compatibility (all memory params optional)
✅ Uses feature flag for safe rollout (`MEMORY_ENABLED`)
✅ Provides transparent context augmentation
✅ Scales with existing infrastructure (PostgreSQL + PGVector)
✅ Supports GDPR compliance and privacy controls

**Key Advantage over V1:** Seamless integration with existing frontend - no endpoint URL changes, no breaking changes, simple parameter additions.
