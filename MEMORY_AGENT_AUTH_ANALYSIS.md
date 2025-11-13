# Authentication Analysis & Final Memory Agent Design

## Key Findings from Code Review

### 1. Existing Authentication Mechanism ✅

**File:** `/tmp/denodo-ai-sdk/api/utils/sdk_utils.py` (Lines 435-444)

```python
def authenticate(
    basic_credentials: Annotated[HTTPBasicCredentials, Depends(security_basic)],
    bearer_credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_bearer)]
):
    if bearer_credentials is not None:
        return bearer_credentials.credentials  # Returns token string
    elif basic_credentials is not None:
        return (basic_credentials.username, basic_credentials.password)  # Returns tuple
    else:
        raise HTTPException(status_code=401, detail="Authentication required")
```

**Returns:**
- **Basic Auth:** `(username, password)` tuple
- **Bearer Auth:** Token string

### 2. answerQuestion Endpoint Usage

**File:** `/tmp/denodo-ai-sdk/api/endpoints/answerQuestion.py` (Line 119-121)

```python
@router.post('/answerQuestion', ...)
async def answer_question_post(
    endpoint_request: answerQuestionRequest,
    auth: str = Depends(authenticate),  # ← Gets (username, password) or token
):
```

The `auth` parameter is **already available** in every answerQuestion call!

### 3. Sample Chatbot Implementation

**File:** `/tmp/denodo-ai-sdk/sample_chatbot/main.py`

**User Login (Lines 333-368):**
```python
@chatbot_bp.route('/signin', methods=['POST'])
def signin():
    username = data.get('username')
    password = data.get('password')

    # Validate credentials by calling AI SDK
    status, response_data = get_user_views(
        api_host=AI_SDK_HOST,
        username=username,
        password=password,
        ...
    )

    # Create user session
    user = User(username, password)  # username stored as user.id
    users[username] = user
    login_user(user)
```

**Making API Calls (Lines 234-237):**
```python
"answer_question": {
    "function": answer_question,
    "params": {
        "username": self.id,  # ← Username from login
        "password": self.password,
        ...
    }
}
```

**File:** `/tmp/denodo-ai-sdk/sample_chatbot/chatbot_tools.py` (Lines 76-77)

```python
endpoint = f'{api_host}/answerQuestion'
response = make_ai_sdk_request(endpoint, request_body, (username, password), ...)
```

**File:** `/tmp/denodo-ai-sdk/sample_chatbot/chatbot_utils.py` (Lines 520-523)

```python
response = requests.post(
    endpoint,
    json=payload,
    auth=auth_tuple,  # ← HTTP Basic Auth: (username, password)
    ...
)
```

---

## 🎯 REVISED DESIGN: Zero-Parameter Memory Agent

### Key Insight
**We don't need to add ANY new parameters!** The `user_id` can be extracted from the existing `auth` parameter.

### Implementation Strategy

#### Step 1: Extract user_id from auth

**File:** `/tmp/denodo-ai-sdk/api/endpoints/answerQuestion.py`

```python
async def process_question(request_data: answerQuestionRequest, auth: str):
    # ═══════════════════════════════════════════════════
    # NEW: Extract user_id from auth
    # ═══════════════════════════════════════════════════
    user_id = None

    if isinstance(auth, tuple):
        # Basic Auth: (username, password)
        user_id = auth[0]  # Username is the user_id
    elif isinstance(auth, str):
        # Bearer Token: Extract from JWT or use token hash
        try:
            # Option 1: Decode JWT to get user claim
            import jwt
            decoded = jwt.decode(auth, options={"verify_signature": False})
            user_id = decoded.get('sub') or decoded.get('username') or decoded.get('email')
        except:
            # Option 2: Use hash of token as user_id (privacy-preserving)
            import hashlib
            user_id = hashlib.sha256(auth.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════
    # NEW: Memory Processing (if user_id available)
    # ═══════════════════════════════════════════════════
    memory_enabled = (
        user_id is not None and
        os.getenv("MEMORY_ENABLED", "0") == "1"
    )

    augmented_question = request_data.question
    conversation_metadata = None
    memory_context_used = None

    if memory_enabled:
        try:
            memory_manager = state_manager.get_memory_manager()

            # 1. Get/create conversation (auto-generated session_id)
            conversation = memory_manager.get_or_create_conversation(
                user_id=user_id,
                session_id=None,  # Auto-generate based on time window
                vdp_database_names=request_data.vdp_database_names.split(',') if request_data.vdp_database_names else []
            )

            # 2. Load recent history
            history = memory_manager.get_conversation_history(
                conversation_id=conversation.id,
                max_messages=10
            )

            # 3. Semantic search
            similar_memories = memory_manager.semantic_search_memory(
                user_id=user_id,
                query=request_data.question,
                k=3
            )

            # 4. Augment question with context
            if history or similar_memories:
                augmented_question = memory_manager.build_augmented_context(
                    current_question=request_data.question,
                    history=history,
                    similar_memories=similar_memories
                )

            memory_context_used = {
                "history_messages_count": len(history),
                "similar_queries_count": len(similar_memories),
                "memory_enabled": True
            }

        except Exception as e:
            logger.error(f"Memory error: {e}")
            memory_enabled = False

    # ═══════════════════════════════════════════════════
    # EXISTING: answerQuestion logic (uses augmented_question)
    # ═══════════════════════════════════════════════════
    # ... rest of existing code ...
```

---

## Comparison: V2 vs Final Design

| Aspect | V2 (Optional user_id param) | FINAL (Extract from auth) ✅ |
|--------|----------------------------|------------------------------|
| **API Changes** | Add optional user_id parameter | Zero API changes |
| **Frontend Changes** | Add user_id when ready | Zero frontend changes |
| **User Identification** | Manual parameter | Automatic from auth |
| **Deployment Risk** | Low | Minimal |
| **Backward Compatibility** | Yes (optional param) | Perfect (no changes) |
| **Implementation Complexity** | Low | Very Low |

---

## Benefits of Final Design

✅ **Completely Transparent** - Zero changes to API contract
✅ **Zero Frontend Work** - Existing chatbot works unchanged
✅ **Automatic User Tracking** - Extracts from existing auth
✅ **Privacy-Preserving** - Option to hash Bearer tokens
✅ **Feature Flag Control** - `MEMORY_ENABLED=0/1` to toggle
✅ **Gradual Rollout** - Enable/disable per environment

---

## Session Management Strategy

### Option 1: Time-Based Auto-Grouping (Recommended)

**Concept:** Automatically group questions within a time window into the same conversation.

```python
def get_or_create_conversation(self, user_id: str, vdp_database_names: List[str]):
    """
    Get active conversation or create new one.
    - Active = last updated within TIME_WINDOW (e.g., 30 minutes)
    - If no active conversation, create new one
    """
    TIME_WINDOW_MINUTES = 30

    # Find conversations updated in last 30 minutes
    active_conversation = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.updated_at >= datetime.utcnow() - timedelta(minutes=TIME_WINDOW_MINUTES)
    ).order_by(Conversation.updated_at.desc()).first()

    if active_conversation:
        return active_conversation
    else:
        # Create new conversation
        return Conversation(
            user_id=user_id,
            vdp_database_names=vdp_database_names,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
```

**Behavior:**
- User asks: "Show me sales 2024" → Creates conversation #1
- 5 minutes later: "What about 2023?" → Uses conversation #1 (within 30 min window)
- 45 minutes later: "Show me customers" → Creates conversation #2 (new topic, time gap)

### Option 2: Explicit Session Reset (Frontend-Triggered)

Add a "New Conversation" button in frontend that sends a special header:

```python
# Frontend adds header
headers = {
    "X-New-Conversation": "true"
}

# Backend checks header
new_conversation = request.headers.get("X-New-Conversation") == "true"
```

**Recommendation:** Use **Option 1** (time-based) because it requires zero frontend changes.

---

## Response Format (Unchanged API Contract)

**Before Memory:**
```json
{
    "answer": "...",
    "sql_query": "...",
    "tables_used": [...],
    // ... existing fields
}
```

**After Memory (when MEMORY_ENABLED=1):**
```json
{
    "answer": "...",
    "sql_query": "...",
    "tables_used": [...],
    // ... existing fields (unchanged)

    // NEW optional fields (only present if memory used)
    "conversation_metadata": {
        "conversation_id": "uuid",
        "message_count": 5,
        "session_age_minutes": 12,
        "expires_at": "2025-11-20T10:00:00Z"
    },
    "memory_context_used": {
        "history_messages_count": 4,
        "similar_queries_count": 2,
        "memory_enabled": true
    }
}
```

---

## Configuration

**Add to `sdk_config.env`:**

```bash
##################################################################################
# Section 8: Conversation Memory
##################################################################################

# Enable memory feature (0 = disabled, 1 = enabled)
MEMORY_ENABLED = 1

# PostgreSQL connection for conversation storage
MEMORY_DB_CONNECTION_STRING = postgresql+psycopg://user:pass@localhost:5432/denodo_ai_memory

# Vector store for semantic memory (reuse PGVector from metadata)
MEMORY_VECTOR_STORE = pgvector
MEMORY_VECTOR_COLLECTION_NAME = ai_sdk_conversation_memory

# Retention period (days)
MEMORY_RETENTION_DAYS = 7

# Conversation auto-grouping window (minutes)
# Questions within this window are grouped into same conversation
MEMORY_SESSION_WINDOW_MINUTES = 30

# Context retrieval settings
MEMORY_MAX_HISTORY_MESSAGES = 10
MEMORY_SEMANTIC_SEARCH_K = 3
MEMORY_MAX_CONTEXT_TOKENS = 4000

# Cleanup interval (hours)
MEMORY_CLEANUP_INTERVAL_HOURS = 24
```

---

## Implementation Steps

### Phase 1: Database Setup
1. Create PostgreSQL database
2. Run schema creation script (users, conversations, messages, embeddings)
3. Test connection

### Phase 2: Memory Manager
1. Implement `MemoryManager` class
2. Conversation CRUD operations
3. Context augmentation logic
4. Unit tests

### Phase 3: Integration
1. Modify `answerQuestion.py` to extract user_id from auth
2. Add memory pre/post processing
3. Test with sample chatbot
4. No frontend changes needed!

### Phase 4: Deployment
1. Set `MEMORY_ENABLED = 0` initially
2. Deploy to staging
3. Enable memory: `MEMORY_ENABLED = 1`
4. Monitor logs
5. Production rollout

---

## Testing Scenarios

### Test 1: Basic Auth User (Sample Chatbot)
```python
# User logs in with username="john@example.com", password="secret"
# Makes multiple questions
response1 = client.post("/answerQuestion",
    json={"question": "Show me sales 2024", ...},
    auth=("john@example.com", "secret")
)
# Memory tracks conversation for user_id="john@example.com"

response2 = client.post("/answerQuestion",
    json={"question": "What about 2023?", ...},
    auth=("john@example.com", "secret")
)
# Uses context from response1
```

### Test 2: Bearer Token User
```python
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # JWT with sub="user123"
response = client.post("/answerQuestion",
    json={"question": "Show me data", ...},
    headers={"Authorization": f"Bearer {token}"}
)
# Memory tracks conversation for user_id="user123" (extracted from JWT)
```

### Test 3: Memory Disabled
```bash
# Set MEMORY_ENABLED=0 in config
response = client.post("/answerQuestion", ...)
# Response has no conversation_metadata field
# Existing behavior unchanged
```

---

## Summary

### What Changed from V2 → Final
- ❌ **Removed:** `user_id`, `session_id`, `use_memory` request parameters
- ✅ **Added:** Automatic user_id extraction from existing `auth` parameter
- ✅ **Result:** **ZERO API contract changes!**

### Frontend Impact
- **V1 Design:** Must call new `/memoryChat` endpoint
- **V2 Design:** Must add `user_id` parameter
- **Final Design:** **NO CHANGES NEEDED!** 🎉

### Deployment Risk
- **V1:** High (new endpoint, route changes)
- **V2:** Low (optional parameters)
- **Final:** **Minimal** (feature flag only)

### Next Steps
1. ✅ Approve this final design
2. Implement database schema
3. Build MemoryManager class
4. Integrate into answerQuestion endpoint
5. Test with sample chatbot (zero changes required!)
6. Deploy with `MEMORY_ENABLED=0`, then enable gradually

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Sample Chatbot Frontend (ZERO CHANGES)                     │
│ - User logs in: username/password                          │
│ - Calls /answerQuestion with Basic Auth                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Basic Auth: (username, password)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ /answerQuestion Endpoint                                    │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ 1. auth = Depends(authenticate)                       │   │
│ │    → Returns (username, password) tuple               │   │
│ │                                                         │   │
│ │ 2. Extract user_id:                                   │   │
│ │    if isinstance(auth, tuple):                        │   │
│ │        user_id = auth[0]  # username                  │   │
│ │    elif isinstance(auth, str):                        │   │
│ │        user_id = jwt_decode(auth)['sub']              │   │
│ │                                                         │   │
│ │ 3. if MEMORY_ENABLED and user_id:                     │   │
│ │    ├─ Load conversation history                       │   │
│ │    ├─ Semantic search similar queries                 │   │
│ │    ├─ Augment question with context                   │   │
│ │    ├─ Process with existing answerQuestion logic      │   │
│ │    └─ Store interaction in memory                     │   │
│ │    else:                                              │   │
│ │        Original behavior (no memory)                  │   │
│ └───────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ PostgreSQL   │  │ PGVector     │  │ Data Catalog │
│ Conversations│  │ Semantic     │  │ (VQL Exec)   │
│ & Messages   │  │ Memory       │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Perfect for production! Zero-touch deployment! 🚀**
