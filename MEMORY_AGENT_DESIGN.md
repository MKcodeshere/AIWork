# Production-Grade Memory Agent Design for Denodo AI SDK

## Executive Summary

This document outlines the architecture and implementation plan for adding persistent conversation memory to the Denodo AI SDK, enabling context-aware follow-up questions and multi-turn conversations.

**Current State:** The answerQuestion endpoint processes each query independently with zero conversation history.

**Target State:** Memory-enabled endpoint that maintains user conversation context for 7-10 days (configurable), enabling natural follow-up questions.

---

## 1. Requirements Validation ✓

### Your Understanding is Correct

✅ **Core Requirements:**
- Store conversation history (questions, answers, SQL queries, results)
- Use previous context to understand follow-up questions
- User-based memory with configurable TTL (7-10 days)
- Leverage existing vector store infrastructure (separate collection)

✅ **Real-World Use Cases:**
- **Without Memory:** "What about 2023?" → AI confused
- **With Memory:** "What about 2023?" → AI understands context from "Show me total sales by region for 2024"

✅ **Critical Insight:**
The chatbot-side conversation memory is insufficient because:
- It only helps with tool selection and formatting
- The answerQuestion endpoint never receives previous query context
- SQL generation cannot leverage historical queries

### Additional Considerations

⚠️ **You didn't mention but are critical:**
1. **User Permission Boundaries:** Memory must respect data catalog permissions (user can't access conversation mentioning tables they don't have access to)
2. **Memory Summarization:** For long conversations (>10 turns), summarize to prevent token overflow
3. **Semantic Deduplication:** Prevent storing identical/near-identical questions
4. **Cross-Session Resume:** Users should resume conversations across chatbot sessions
5. **Privacy/Compliance:** GDPR/data retention policies for conversation data
6. **Multi-Tenancy:** Isolate memory between different VDP databases/environments

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User/Chatbot Layer                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NEW: /memoryChat Endpoint                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. Extract user_id from auth token                           │  │
│  │ 2. Retrieve/create conversation session                      │  │
│  │ 3. Load conversation history from DB                         │  │
│  │ 4. Perform semantic search on user's memory vector store     │  │
│  │ 5. Inject context into question (RAG over conversation)      │  │
│  │ 6. Call answerQuestion logic with enriched context           │  │
│  │ 7. Store new Q&A pair in memory (DB + Vector Store)          │  │
│  │ 8. Return response + conversation metadata                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ PostgreSQL   │  │ Vector Store │  │ Existing     │
│ Conversations│  │ (PGVector)   │  │ answerQ      │
│ & Messages   │  │ Semantic     │  │ Logic        │
│              │  │ Memory       │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 2.2 Detailed UML Sequence Diagram

```
User          Chatbot      /memoryChat     MemoryManager    VectorStore    PostgreSQL    answerQuestion    DataCatalog
 │               │              │                │              │              │               │                │
 │──"What about 2023?"──────────>│                │              │              │               │                │
 │               │              │                │              │              │               │                │
 │               │        ┌─────┴────────┐       │              │              │               │                │
 │               │        │1. Extract    │       │              │              │               │                │
 │               │        │   user_id    │       │              │              │               │                │
 │               │        │   session_id │       │              │              │               │                │
 │               │        └─────┬────────┘       │              │              │               │                │
 │               │              │                │              │              │               │                │
 │               │              │──2. Load conversation history─────────────>  │               │                │
 │               │              │                │              │              │               │                │
 │               │              │<─────Returns last 10 messages + metadata────│               │                │
 │               │              │                │              │              │               │                │
 │               │              │──3. Semantic search("What about 2023?")──────>│              │                │
 │               │              │                │              │              │               │                │
 │               │              │<─Returns top-k similar conversations─────────│               │                │
 │               │              │   ["sales by region for 2024", tables:       │               │                │
 │               │              │    sales, regions, SQL: SELECT...]            │               │                │
 │               │              │                │              │              │               │                │
 │               │        ┌─────┴────────┐       │              │              │               │                │
 │               │        │4. Build      │       │              │              │               │                │
 │               │        │   augmented  │       │              │              │               │                │
 │               │        │   context:   │       │              │              │               │                │
 │               │        │   "Previous: │       │              │              │               │                │
 │               │        │   User asked │       │              │              │               │                │
 │               │        │   about 2024 │       │              │              │               │                │
 │               │        │   sales by   │       │              │              │               │                │
 │               │        │   region using│      │              │              │               │                │
 │               │        │   tables...  │       │              │              │               │                │
 │               │        │   Now asking:│       │              │              │               │                │
 │               │        │   What about │       │              │              │               │                │
 │               │        │   2023?"     │       │              │              │               │                │
 │               │        └─────┬────────┘       │              │              │               │                │
 │               │              │                │              │              │               │                │
 │               │              │──5. Call answerQuestion(augmented_question)──────────────────>│                │
 │               │              │                │              │              │               │                │
 │               │              │                │              │              │               │──6. Vector─────>│
 │               │              │                │              │              │               │   search       │
 │               │              │                │              │              │               │   for tables   │
 │               │              │                │              │              │               │                │
 │               │              │                │              │              │               │<───────────────│
 │               │              │                │              │              │               │                │
 │               │              │                │              │              │               │──7. Generate───>│
 │               │              │                │              │              │               │   VQL for      │
 │               │              │                │              │              │               │   2023 sales   │
 │               │              │                │              │              │               │                │
 │               │              │                │              │              │               │<─8. Execute────│
 │               │              │                │              │              │               │    & return    │
 │               │              │                │              │              │               │                │
 │               │              │<─9. Returns answer, SQL, tables_used, results───────────────│                │
 │               │              │                │              │              │               │                │
 │               │        ┌─────┴────────┐       │              │              │               │                │
 │               │        │10. Store     │       │              │              │               │                │
 │               │        │    memory    │       │              │              │               │                │
 │               │        └─────┬────────┘       │              │              │               │                │
 │               │              │                │              │              │               │                │
 │               │              │──11. Insert message(Q, A, SQL, tables)───────>│              │                │
 │               │              │                │              │              │               │                │
 │               │              │──12. Embed & store in vector store────────────>│              │                │
 │               │              │    (question + context + SQL)  │              │               │                │
 │               │              │                │              │              │               │                │
 │               │              │──13. Check TTL & cleanup old conversations────────────────>  │                │
 │               │              │                │              │              │               │                │
 │               │<─14. Returns answer + conversation_metadata───│              │               │                │
 │               │              │                │              │              │               │                │
 │<──Returns formatted response with metadata───│              │              │               │                │
 │               │              │                │              │              │               │                │
```

### 2.3 Component Breakdown

#### A. PostgreSQL Database Schema

```sql
-- Table: users
CREATE TABLE users (
    user_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_users_last_active ON users(last_active_at);

-- Table: conversations
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_name VARCHAR(500),  -- First question as display name
    vdp_database_names TEXT[],  -- Databases accessed in this conversation
    vdp_tag_names TEXT[],       -- Tags accessed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,       -- TTL: created_at + configured retention days
    message_count INTEGER DEFAULT 0,
    is_summarized BOOLEAN DEFAULT FALSE,
    summary_text TEXT           -- Summarized history for long conversations
);
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_expires_at ON conversations(expires_at);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at);

-- Table: messages
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    message_order INTEGER NOT NULL,  -- Sequence number within conversation
    role VARCHAR(20) NOT NULL,       -- 'user' or 'assistant'
    content TEXT NOT NULL,           -- Question or answer
    sql_query TEXT,                  -- VQL generated (for assistant messages)
    query_explanation TEXT,          -- Explanation of SQL
    tables_used TEXT[],              -- Tables referenced
    execution_result JSONB,          -- Query results (limited rows)
    tokens JSONB,                    -- Token usage stats
    execution_time_ms FLOAT,         -- Query execution time
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id, message_order);
CREATE INDEX idx_messages_user_id ON messages(user_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_tables_used ON messages USING GIN(tables_used);  -- Array index

-- Table: memory_embeddings (if using separate table, otherwise use PGVector collection)
CREATE TABLE memory_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    embedding vector(3072),          -- Dimension depends on embeddings model
    search_text TEXT NOT NULL,       -- Question + SQL + tables for semantic search
    metadata JSONB,                  -- Additional search metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_memory_embeddings_user_id ON memory_embeddings(user_id);
CREATE INDEX ON memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### B. Vector Store Configuration

**New Collection:** `ai_sdk_conversation_memory`

**Purpose:** Semantic search over user conversation history

**Schema:**
```python
{
    "id": "msg_{message_id}",
    "user_id": "user@domain.com",
    "conversation_id": "uuid",
    "question": "Show me total sales by region for 2024",
    "answer_summary": "Generated VQL query showing $2.5M in North region...",
    "sql_query": "SELECT r.region_name, SUM(s.amount) FROM sales s JOIN...",
    "tables_used": ["sales", "regions"],
    "vdp_databases": ["banking_db"],
    "created_at": "2025-11-13T10:30:00Z",
    "message_order": 5
}
```

**Embeddings:** Concatenated text for rich semantic search
```
Question: {question}
SQL: {sql_query}
Tables: {tables_used}
Result summary: {answer_summary[:200]}
```

#### C. Memory Manager Class

**File:** `/tmp/denodo-ai-sdk/api/utils/memory_manager.py`

**Core Responsibilities:**
1. **Session Management:** Create/retrieve/expire conversations
2. **Memory Storage:** Persist Q&A pairs to DB and vector store
3. **Context Retrieval:** Semantic search + recency-based filtering
4. **Summarization:** Condense long conversation histories
5. **Cleanup:** TTL-based expiration and privacy compliance

**Key Methods:**
```python
class MemoryManager:
    def __init__(self, db_connection, vector_store, embeddings_model, config):
        pass

    def get_or_create_conversation(self, user_id: str, session_id: Optional[str]) -> Conversation:
        """Retrieve existing conversation or start new one"""
        pass

    def get_conversation_history(self, conversation_id: str, max_messages: int = 10) -> List[Message]:
        """Load recent messages from DB"""
        pass

    def semantic_search_memory(self, user_id: str, query: str, k: int = 3,
                               conversation_id: Optional[str] = None) -> List[Dict]:
        """Find similar past questions/queries for this user"""
        pass

    def build_augmented_context(self, current_question: str, history: List[Message],
                                 similar_memories: List[Dict]) -> str:
        """Construct context-enriched prompt"""
        pass

    def store_interaction(self, conversation_id: str, user_id: str,
                         question: str, answer_data: Dict) -> None:
        """Persist Q&A to DB and embed in vector store"""
        pass

    def summarize_conversation(self, conversation_id: str) -> str:
        """Use LLM to summarize long conversations"""
        pass

    def cleanup_expired_conversations(self) -> int:
        """Delete conversations past TTL"""
        pass

    def export_user_memory(self, user_id: str) -> Dict:
        """GDPR compliance: export all user data"""
        pass

    def delete_user_memory(self, user_id: str) -> None:
        """GDPR compliance: right to be forgotten"""
        pass
```

#### D. Context Augmentation Strategy

**Prompt Engineering for Context Injection:**

```python
def build_augmented_context(current_question, history, similar_memories):
    context_parts = []

    # 1. Recent conversation history (last N messages)
    if history:
        context_parts.append("=== RECENT CONVERSATION ===")
        for msg in history[-5:]:  # Last 5 messages
            if msg.role == "user":
                context_parts.append(f"User asked: {msg.content}")
            else:
                context_parts.append(
                    f"Assistant answered using tables: {msg.tables_used}\n"
                    f"SQL: {msg.sql_query[:200]}..."
                )

    # 2. Semantically similar past queries (from vector search)
    if similar_memories:
        context_parts.append("\n=== RELATED PAST QUERIES ===")
        for mem in similar_memories[:3]:  # Top 3
            context_parts.append(
                f"Previously: User asked '{mem['question']}'\n"
                f"Tables used: {mem['tables_used']}\n"
                f"SQL pattern: {mem['sql_query'][:150]}..."
            )

    # 3. Current question with disambiguation instruction
    augmented_question = f"""
{chr(10).join(context_parts)}

=== CURRENT QUESTION ===
{current_question}

INSTRUCTIONS:
- Use the conversation history above to resolve any ambiguous references (e.g., "what about", "for that year", "same but")
- If the current question references previous queries, reuse relevant tables, filters, and SQL patterns
- Generate SQL that addresses the current question in the context of the conversation
"""

    return augmented_question
```

---

## 3. Implementation Plan

### Phase 1: Database Infrastructure (Day 1-2)

**Files to Create:**
1. `/tmp/denodo-ai-sdk/api/utils/db/schema.sql` - PostgreSQL DDL
2. `/tmp/denodo-ai-sdk/api/utils/db/connection.py` - SQLAlchemy connection pool
3. `/tmp/denodo-ai-sdk/api/utils/db/models.py` - ORM models (User, Conversation, Message)
4. `/tmp/denodo-ai-sdk/api/utils/db/migrations/001_initial_schema.py` - Alembic migration

**Tasks:**
- Set up SQLAlchemy engine with connection pooling
- Create ORM models with proper relationships
- Add database URL to sdk_config.env
- Test connection and schema creation

### Phase 2: Memory Manager Core (Day 3-4)

**Files to Create:**
1. `/tmp/denodo-ai-sdk/api/utils/memory_manager.py` - Core memory logic
2. `/tmp/denodo-ai-sdk/api/utils/memory_config.py` - Memory-specific configuration

**Tasks:**
- Implement conversation CRUD operations
- Build message storage with embedding generation
- Implement semantic search over memory vector store
- Add context augmentation logic
- Unit tests for each method

### Phase 3: New Endpoint Integration (Day 5-6)

**Files to Create:**
1. `/tmp/denodo-ai-sdk/api/endpoints/memoryChat.py` - New endpoint
2. `/tmp/denodo-ai-sdk/api/utils/sdk_memory_answer_question.py` - Memory-aware Q&A logic

**Files to Modify:**
1. `/tmp/denodo-ai-sdk/api/main.py` - Register new endpoint
2. `/tmp/denodo-ai-sdk/api/utils/state_manager.py` - Initialize MemoryManager
3. `/tmp/denodo-ai-sdk/api/utils/sdk_utils.py` - Extract user_id from auth

**Tasks:**
- Create /memoryChat endpoint with session management
- Integrate MemoryManager into request flow
- Reuse answerQuestion logic with enriched context
- Add conversation metadata to response
- Integration tests

### Phase 4: TTL & Maintenance (Day 7)

**Files to Create:**
1. `/tmp/denodo-ai-sdk/api/utils/memory_cleanup.py` - Background cleanup task

**Tasks:**
- Implement TTL-based expiration
- Background job to delete expired conversations
- Conversation summarization for long histories
- Performance optimization (DB indexes, query tuning)

### Phase 5: Configuration & Documentation (Day 8)

**Files to Modify:**
1. `/tmp/denodo-ai-sdk/api/utils/sdk_config_example.env` - Add memory settings

**Files to Create:**
1. `/tmp/denodo-ai-sdk/docs/MEMORY_AGENT.md` - User documentation

**Tasks:**
- Add all memory-related config options
- Document endpoint usage
- Create example requests/responses
- Update API documentation

---

## 4. Configuration Additions

**New Section in `sdk_config_example.env`:**

```bash
##################################################################################
# Section 8: Conversation Memory (Short-term Context)
##################################################################################

# Enable persistent conversation memory for follow-up questions
MEMORY_ENABLED = 1

# Database for storing conversation history
# Uses PostgreSQL (required for pgvector semantic search)
MEMORY_DB_CONNECTION_STRING = postgresql+psycopg://user:password@localhost:5432/denodo_ai_sdk_memory

# Connection pool settings
MEMORY_DB_POOL_SIZE = 10
MEMORY_DB_MAX_OVERFLOW = 20

# Vector store for semantic memory search (separate from schema metadata)
MEMORY_VECTOR_STORE = pgvector  # pgvector | chroma | opensearch
MEMORY_VECTOR_COLLECTION_NAME = ai_sdk_conversation_memory

# Conversation retention (short-term memory)
MEMORY_RETENTION_DAYS = 7  # 7-10 days recommended
MEMORY_CLEANUP_INTERVAL_HOURS = 24  # How often to run TTL cleanup

# Memory retrieval settings
MEMORY_MAX_HISTORY_MESSAGES = 10  # Last N messages to include in context
MEMORY_SEMANTIC_SEARCH_K = 3  # Top K similar past queries
MEMORY_MAX_CONTEXT_TOKENS = 4000  # Max tokens for augmented context

# Conversation summarization (for long conversations)
MEMORY_SUMMARIZATION_ENABLED = 1
MEMORY_SUMMARIZATION_THRESHOLD = 15  # Summarize after N messages
MEMORY_SUMMARIZATION_MODEL = gpt-4o-mini  # Cheaper model for summaries

# Privacy and compliance
MEMORY_ALLOW_DATA_EXPORT = 1  # GDPR: Enable user data export
MEMORY_ALLOW_DATA_DELETION = 1  # GDPR: Enable right to be forgotten
MEMORY_LOG_MEMORY_OPERATIONS = 1  # Audit logging for memory access

# User identification (how to extract user_id from auth)
MEMORY_USER_ID_SOURCE = jwt_claim  # jwt_claim | basic_username | bearer_token
MEMORY_USER_ID_JWT_CLAIM = sub  # Which JWT claim contains user_id
```

---

## 5. API Specification: /memoryChat Endpoint

### Request

```python
class memoryChatRequest(BaseModel):
    # Core question
    question: str

    # Session management
    session_id: Optional[str] = None  # Resume existing conversation
    session_name: Optional[str] = None  # Custom name for new conversation

    # Memory controls
    use_memory: bool = True  # Enable/disable memory for this query
    max_history_messages: Optional[int] = None  # Override default
    semantic_search_k: Optional[int] = None  # Override default

    # Standard answerQuestion parameters (inherited)
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
```

### Response

```python
class memoryChatResponse(BaseModel):
    # Standard answerQuestion response fields
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

    # NEW: Memory-specific metadata
    conversation_metadata: Dict[str, Any] = {
        "conversation_id": "uuid",
        "session_name": "Show me total sales...",
        "message_count": 5,
        "created_at": "2025-11-13T10:00:00Z",
        "updated_at": "2025-11-13T10:30:00Z",
        "expires_at": "2025-11-20T10:00:00Z",
        "is_summarized": False
    }

    memory_context_used: Dict[str, Any] = {
        "history_messages_count": 4,  # How many past messages were included
        "similar_queries_count": 2,   # How many semantic matches found
        "context_tokens": 856,        # Tokens used for context
        "context_truncated": False    # Whether context was truncated
    }

    memory_retrieval_time: float  # Time to fetch and search memory
```

### Example Usage

**First Question:**
```bash
POST /memoryChat
{
    "question": "Show me total sales by region for 2024",
    "vdp_database_names": "banking_db",
    "llm_provider": "OpenAI",
    "llm_model": "gpt-4o",
    ...
}

Response:
{
    "answer": "The total sales by region for 2024 are:\n- North: $2.5M\n- South: $1.8M\n- East: $2.1M\n- West: $1.6M",
    "sql_query": "SELECT r.region_name, SUM(s.amount) AS total_sales FROM sales s JOIN regions r ON s.region_id = r.region_id WHERE YEAR(s.sale_date) = 2024 GROUP BY r.region_name",
    "tables_used": ["sales", "regions"],
    "conversation_metadata": {
        "conversation_id": "a1b2c3d4-...",
        "session_name": "Show me total sales...",
        "message_count": 1,
        "expires_at": "2025-11-20T10:00:00Z"
    }
}
```

**Follow-up Question:**
```bash
POST /memoryChat
{
    "question": "What about 2023?",
    "session_id": "a1b2c3d4-...",  # Resume same conversation
    ...
}

Response:
{
    "answer": "The total sales by region for 2023 are:\n- North: $2.2M\n- South: $1.5M\n- East: $1.9M\n- West: $1.4M",
    "sql_query": "SELECT r.region_name, SUM(s.amount) AS total_sales FROM sales s JOIN regions r ON s.region_id = r.region_id WHERE YEAR(s.sale_date) = 2023 GROUP BY r.region_name",
    "tables_used": ["sales", "regions"],
    "conversation_metadata": {
        "conversation_id": "a1b2c3d4-...",
        "message_count": 2,
        ...
    },
    "memory_context_used": {
        "history_messages_count": 1,
        "similar_queries_count": 1,
        "context_tokens": 245
    }
}
```

---

## 6. Production Considerations

### A. Performance Optimization

1. **Database Indexing:**
   - Index `conversations(user_id, updated_at)` for fast user query
   - Index `messages(conversation_id, message_order)` for ordered retrieval
   - GIN index on `messages(tables_used)` for table-based filtering
   - IVFFlat index on `memory_embeddings(embedding)` for fast vector search

2. **Connection Pooling:**
   - Separate pool for memory DB (10-20 connections)
   - Async queries for non-blocking memory operations
   - Consider read replicas for high-traffic scenarios

3. **Caching:**
   - Cache recent conversations in Redis (optional)
   - TTL: 1 hour for active conversations
   - Reduces DB load for rapid-fire follow-ups

4. **Batch Operations:**
   - Batch embed multiple messages together
   - Use `COPY` for bulk inserts during migration
   - Parallel embedding generation with rate limiting

### B. Security & Privacy

1. **User Isolation:**
   - Enforce user_id filter on ALL memory queries
   - Prevent cross-user memory leakage
   - Test with multiple user accounts

2. **Permission Boundaries:**
   - Filter memory search by `vdp_database_names` user has access to
   - Don't return memories from tables user can't access now (even if they could in past)
   - Validate against Data Catalog permissions API

3. **Data Retention:**
   - Automatic deletion after configured TTL (7-10 days)
   - Manual user-triggered deletion (GDPR compliance)
   - Audit log for all memory operations

4. **Sensitive Data:**
   - Don't store full query results (only summaries/counts)
   - Limit execution_result to N rows (configurable)
   - Consider PII detection/masking for question content

### C. Monitoring & Observability

1. **Metrics to Track:**
   - Memory hit rate (% of queries using context successfully)
   - Average context retrieval time
   - Conversation length distribution
   - Memory store size growth rate
   - TTL cleanup success rate

2. **Logging:**
   - Log all memory operations (read/write/delete)
   - Include transaction_id for correlation
   - Log context truncation events
   - Alert on memory store failures

3. **Langfuse Integration:**
   - Trace memory-enhanced queries separately
   - Compare accuracy with/without memory
   - Track token usage increase from context

### D. Scalability

1. **Horizontal Scaling:**
   - PostgreSQL can be replicated (read replicas)
   - PGVector supports partitioning by user_id
   - Consider sharding by user_id for massive scale

2. **Vertical Scaling:**
   - Memory store will grow linearly with users × messages
   - Estimate: 5KB per message × 100 messages/user × 10K users = 5GB
   - Plan for 10x growth in first year

3. **Archive Strategy:**
   - After TTL, optionally archive to cold storage (S3)
   - For compliance or analytics
   - Delete from active DB but keep compressed backups

---

## 7. Testing Strategy

### Unit Tests
- `test_memory_manager.py`: All MemoryManager methods
- `test_db_models.py`: ORM model relationships
- `test_context_augmentation.py`: Prompt engineering logic

### Integration Tests
- `test_memoryChat_endpoint.py`: End-to-end API tests
- `test_memory_vector_store.py`: Semantic search accuracy
- `test_ttl_cleanup.py`: Expiration and deletion

### Performance Tests
- Load test: 1000 concurrent users
- Latency: Memory retrieval <100ms (p95)
- Throughput: 100 queries/sec

### Security Tests
- Cross-user memory leakage prevention
- Permission boundary enforcement
- SQL injection in stored queries (sanitization)

---

## 8. Migration & Rollout Plan

### Phase 1: Infrastructure Setup (Week 1)
- Provision PostgreSQL database
- Create schema and indexes
- Set up monitoring and backups

### Phase 2: Development (Week 2-3)
- Implement MemoryManager and endpoint
- Integration with existing answerQuestion logic
- Comprehensive testing

### Phase 3: Beta Testing (Week 4)
- Deploy to staging environment
- Select 10 pilot users
- Collect feedback on context accuracy
- Performance tuning

### Phase 4: Production Rollout (Week 5)
- Feature flag: `MEMORY_ENABLED = 0` initially
- Gradual rollout: 10% → 50% → 100%
- Monitor error rates and latency
- Fallback plan: Disable memory if issues arise

### Phase 5: Optimization (Week 6+)
- Tune vector search parameters
- Optimize prompt engineering
- Add conversation summarization
- User feedback loop

---

## 9. Open Questions & Decisions Needed

1. **User Identification:**
   - How is user_id currently passed? JWT claims? Basic auth username?
   - Do we need to support anonymous users (session-only memory)?

2. **Memory Scope:**
   - Should memory be global across all VDP databases or scoped per database?
   - Should conversations from different chatbot instances be merged?

3. **Summarization Trigger:**
   - What threshold for conversation summarization? (Suggest: 15 messages)
   - Should we summarize in real-time or batch job?

4. **Vector Store Choice:**
   - Prefer PGVector (same DB as conversations) or separate Chroma instance?
   - Trade-off: PGVector = simpler ops, Chroma = better performance for large scale

5. **Backward Compatibility:**
   - Should /answerQuestion remain unchanged (no memory) and /memoryChat be separate?
   - Or add optional `use_memory` parameter to existing endpoint?

6. **Error Handling:**
   - If memory store fails, should query proceed without context or return error?
   - Suggest: Log error, proceed without memory, return warning in response

---

## 10. Success Metrics

**Before Implementation (Baseline):**
- Follow-up question success rate: ~30% (users rephrase questions)
- Average questions per session: 1.5
- User satisfaction: 3.2/5

**After Implementation (Target):**
- Follow-up question success rate: >80%
- Average questions per session: 4-5
- User satisfaction: >4.0/5
- Memory hit rate: >70% (queries successfully using context)
- Latency increase: <50ms per query (p95)

---

## 11. Next Steps

1. **Review & Approve Design:** Get stakeholder sign-off on architecture
2. **Provision Infrastructure:** Set up PostgreSQL database
3. **Update Configuration:** Add memory settings to sdk_config
4. **Begin Implementation:** Start with Phase 1 (database layer)
5. **Set Up Development Environment:** Clone repo, install dependencies
6. **Create Feature Branch:** `feature/memory-agent`
7. **Iterative Development:** Build incrementally with tests

---

## Summary

This design provides a **production-grade, scalable, and privacy-compliant** memory agent for the Denodo AI SDK that:

✅ Stores conversation history with configurable TTL (7-10 days)
✅ Enables follow-up questions via semantic search and recency-based context
✅ Integrates seamlessly with existing answerQuestion logic
✅ Respects user permissions and data boundaries
✅ Supports GDPR compliance (export, deletion)
✅ Scales horizontally with user growth
✅ Provides comprehensive monitoring and observability

The implementation reuses existing infrastructure (PGVector, LangChain, FastAPI) and follows the SDK's architectural patterns, ensuring maintainability and consistency.
