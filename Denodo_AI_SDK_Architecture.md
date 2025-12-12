# Denodo AI SDK - Complete Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Details](#component-details)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [DeepQuery Architecture](#deepquery-architecture)
6. [Technical Implementation](#technical-implementation)

---

## Overview

The Denodo AI SDK is a Retrieval-Augmented Generation (RAG) system that enables natural language querying of enterprise data through the Denodo Platform. It combines vector search, LangChain, LLMs, and the Denodo Data Catalog to provide accurate, context-aware answers to user questions.

**Core Technology Stack:**
- **Framework**: FastAPI (Python)
- **LLM Integration**: LangChain
- **Vector Databases**: ChromaDB, PGVector, OpenSearch, Pinecone, Qdrant
- **LLM Providers**: OpenAI, Azure, AWS Bedrock, Google Vertex, Anthropic, NVIDIA, Groq, Ollama, Mistral, etc.
- **Data Source**: Denodo Platform via Data Catalog REST API

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER APPLICATION LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Sample Chatbot   │  │   Custom Apps    │  │  Direct API      │          │
│  │   (React UI)     │  │   (External)     │  │   Calls          │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                      │                     │
│           └─────────────────────┴──────────────────────┘                     │
│                                 │                                            │
│                          HTTP/HTTPS Requests                                 │
└─────────────────────────────────┴───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DENODO AI SDK (FastAPI)                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          API ENDPOINTS                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │answerQuestion│  │getMetadata   │  │deepQuery     │               │  │
│  │  │              │  │              │  │              │               │  │
│  │  │streamAnswer  │  │deleteMetadata│  │similarity    │               │  │
│  │  │Question      │  │              │  │Search        │               │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │  │
│  └─────────┼──────────────────┼──────────────────┼───────────────────────┘  │
│            │                  │                  │                           │
│  ┌─────────┴──────────────────┴──────────────────┴───────────────────────┐  │
│  │                    ORCHESTRATION LAYER                                 │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │              sdk_answer_question.py                              │ │  │
│  │  │  • Question Categorization                                       │ │  │
│  │  │  • Query Routing (SQL/Metadata/Data)                            │ │  │
│  │  │  • Execution Flow Management                                     │ │  │
│  │  │  • Error Handling & Retry Logic                                 │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │                sdk_ai_tools.py                                   │ │  │
│  │  │  • query_to_vql()      - SQL Generation                         │ │  │
│  │  │  • query_fixer()       - Error Correction                       │ │  │
│  │  │  • query_reviewer()    - Query Validation                       │ │  │
│  │  │  • answer_view()       - Natural Language Answer Generation     │ │  │
│  │  │  • related_questions() - Suggestion Generation                  │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│            │                  │                  │                           │
│  ┌─────────┼──────────────────┼──────────────────┼───────────────────────┐  │
│  │         │     CORE ABSTRACTION LAYER          │                       │  │
│  │  ┌──────▼────────┐  ┌──────▼────────┐  ┌─────▼──────────┐           │  │
│  │  │  UniformLLM   │  │UniformVector  │  │ UniformEmbed   │           │  │
│  │  │               │  │    Store      │  │   dings        │           │  │
│  │  │ • OpenAI     │  │ • ChromaDB    │  │ • OpenAI       │           │  │
│  │  │ • Azure      │  │ • PGVector    │  │ • Azure        │           │  │
│  │  │ • Bedrock    │  │ • Pinecone    │  │ • Bedrock      │           │  │
│  │  │ • Google     │  │ • OpenSearch  │  │ • Google       │           │  │
│  │  │ • Anthropic  │  │ • Qdrant      │  │ • Ollama       │           │  │
│  │  │ • Ollama     │  │               │  │ • Mistral      │           │  │
│  │  │ • etc.       │  │               │  │                │           │  │
│  │  └───────────────┘  └───────────────┘  └────────────────┘           │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│            │                  │                  │                           │
└────────────┼──────────────────┼──────────────────┼───────────────────────────┘
             │                  │                  │
             ▼                  ▼                  │
┌──────────────────────┐  ┌──────────────────────┐│
│   LLM PROVIDERS      │  │  VECTOR DATABASES    ││
│                      │  │                      ││
│  • OpenAI API        │  │  • ChromaDB (local)  ││
│  • Azure OpenAI      │  │  • PGVector          ││
│  • AWS Bedrock       │  │  • Pinecone          ││
│  • Google Vertex     │  │  • OpenSearch        ││
│  • Anthropic API     │  │  • Qdrant            ││
│  • NVIDIA NIM        │  │                      ││
│  • Groq              │  │                      ││
│  • Ollama (local)    │  │                      ││
└──────────────────────┘  └──────────────────────┘│
                                                   │
             ┌─────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DENODO PLATFORM (Data Source)                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Denodo Data Catalog                                │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │  REST API Endpoints:                                             │ │  │
│  │  │  • /public/api/askaquestion/data      - Metadata Retrieval       │ │  │
│  │  │  • /public/api/askaquestion/execute   - VQL Execution            │ │  │
│  │  │  • /public/api/views/allowed-ids      - Permission Check         │ │  │
│  │  │  • /public/api/ai-sdk/configuration   - Incremental Updates      │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─────────────────────────────────┴──────────────────────────────────┐    │
│  │                      Denodo VDP Server                              │    │
│  │  • Virtual Data Layer                                               │    │
│  │  • Data Virtualization                                              │    │
│  │  • VQL Query Engine                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────┴──────────────────────────────────┐    │
│  │              Data Sources (Virtual Views)                           │    │
│  │  • Databases    • APIs    • Files    • Cloud Services               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. UniformLLM (LLM Abstraction Layer)

**File**: `utils/uniformLLM.py`

**Purpose**: Provides a unified interface to interact with multiple LLM providers through LangChain.

**Supported Providers**:
- OpenAI, Azure OpenAI, AWS Bedrock, Google Vertex AI
- Anthropic, NVIDIA NIM, Groq, Ollama
- Mistral, SambaNova, OpenRouter

**Key Functions**:
```python
class UniformLLM:
    def __init__(provider_name, model_name, temperature, max_tokens)
    def setup_openai()      # OpenAI API configuration
    def setup_azure()       # Azure OpenAI configuration
    def setup_bedrock()     # AWS Bedrock configuration
    def setup_google()      # Google Vertex AI configuration
    def invoke(prompt)      # Synchronous LLM call
    def ainvoke(prompt)     # Asynchronous LLM call
    def stream(prompt)      # Streaming LLM call
```

**Architecture**:
```
UniformLLM
    │
    ├─► LangChain ChatOpenAI
    ├─► LangChain AzureChatOpenAI
    ├─► LangChain ChatBedrock
    ├─► LangChain ChatVertexAI
    ├─► LangChain ChatAnthropic
    └─► LangChain ChatOllama
```

### 2. UniformVectorStore (Vector Database Abstraction)

**File**: `utils/uniformVectorStore.py`

**Purpose**: Unified interface for vector databases to store and retrieve embeddings of database metadata.

**Supported Vector Stores**:
- ChromaDB (default, local)
- PGVector (PostgreSQL)
- OpenSearch
- Pinecone
- Qdrant

**Key Functions**:
```python
class UniformVectorStore:
    def __init__(provider, embeddings, index_name, rate_limit_rpm)
    def add_views(views, parallel)           # Insert metadata
    def search(query, k, filters)            # Semantic search
    def search_by_vector(vector, k, filters) # Direct vector search
    def delete_by_view_id(view_ids)          # Remove metadata
    def get_last_update(source_type, name)   # Incremental updates
```

**Vector Store Schema**:
```
Document {
    id: "view_id"
    page_content: "Natural language description"
    metadata: {
        view_json: "Complete schema JSON"
        view_id: "123"
        database_name: "database_name"
        view_name: "view_name"
        tag_finance: "1"
        tag_sales: "1"
        last_update: "timestamp_ms"
    }
}
```

### 3. UniformEmbeddings (Embeddings Abstraction)

**File**: `utils/uniformEmbeddings.py`

**Purpose**: Unified interface for generating text embeddings.

**Supported Providers**:
- OpenAI (text-embedding-3-large)
- Azure OpenAI
- AWS Bedrock (Titan embeddings)
- Google Vertex AI
- Ollama (bge-m3)
- Mistral, NVIDIA

**Key Functions**:
```python
class UniformEmbeddings:
    def __init__(provider_name, model_name)
    def embed_documents(texts)    # Batch embedding
    def embed_query(text)          # Single query embedding
    def get_dimensions()           # Get embedding dimensions
```

### 4. Data Catalog Integration

**File**: `utils/data_catalog.py`

**Purpose**: Interface with Denodo Data Catalog REST API.

**Key Functions**:
```python
def get_views_metadata_documents(auth, database_name, tag_name, examples_per_table)
    # Retrieves database schema metadata with sample data

def execute_vql(vql, auth, limit)
    # Executes VQL queries against Denodo Platform

def get_allowed_view_ids(auth)
    # Gets list of views user has permissions to access

def activate_incremental(auth, enabled)
    # Enables incremental metadata sync
```

**API Endpoints Used**:
1. `POST /public/api/askaquestion/data` - Fetch metadata
2. `POST /public/api/askaquestion/execute` - Execute VQL
3. `POST /public/api/views/allowed-identifiers` - Check permissions
4. `POST /public/api/ai-sdk/configuration` - Incremental sync

---

## End-to-End Data Flow

### Phase 1: Metadata Ingestion (Setup Phase)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    METADATA INGESTION FLOW                                  │
└────────────────────────────────────────────────────────────────────────────┘

1. User Request
   │
   GET /getMetadata?vdp_database_names=sales&examples_per_table=10
   │
   ▼
2. AI SDK Endpoint (getMetadata.py)
   │
   ├─► Check authentication
   ├─► Activate incremental sync (if enabled)
   └─► For each database/tag:
       │
       ▼
3. Data Catalog API Call
   │
   POST /public/api/askaquestion/data
   {
     "dataMode": "DATABASE",
     "databaseName": "sales",
     "dataUsage": true,
     "dataUsageConfiguration": {
       "tuplesToUse": 10,
       "samplingMethod": "random"
     }
   }
   │
   ▼
4. Denodo Data Catalog Response
   │
   {
     "viewsDetails": [
       {
         "name": "customer",
         "databaseName": "sales",
         "id": 123,
         "description": "Customer information",
         "schema": [
           {
             "name": "customer_id",
             "type": "int",
             "description": "Unique customer identifier",
             "primaryKey": true
           },
           ...
         ],
         "viewFieldDataList": [
           {
             "fieldName": "customer_id",
             "fieldValues": [1001, 1002, 1003, ...]
           }
         ],
         "associationData": [...]
       }
     ]
   }
   │
   ▼
5. Parse & Transform (parse_metadata_json)
   │
   ├─► Clean None values
   ├─► Format schema
   ├─► Attach sample data to columns
   └─► Build associations
   │
   ▼
6. Generate Embeddings (UniformEmbeddings)
   │
   For each view:
   │
   Natural Language Description:
   "Table: sales.customer
    Description: Customer information
    Columns:
    - customer_id (int, Primary Key): Unique customer identifier
    - name (text): Customer full name
    - email (text): Customer email address
    Sample Data: customer_id=[1001,1002,1003], name=[John,Jane,Bob]"
   │
   ├─► Call embeddings API (OpenAI/Bedrock/etc.)
   ├─► Get vector: [0.123, -0.456, 0.789, ..., 0.234]  (1536 dimensions)
   │
   ▼
7. Store in Vector Database (UniformVectorStore)
   │
   ChromaDB.add({
     id: "123",
     embedding: [0.123, -0.456, ...],
     document: "Natural language description",
     metadata: {
       view_json: "{complete schema JSON}",
       view_id: "123",
       database_name: "sales",
       view_name: "customer",
       tag_finance: "1"
     }
   })
   │
   ▼
8. Response to User
   │
   {
     "db_schema_json": [{...}],
     "db_schema_text": ["Table: sales.customer..."],
     "vdb_list": ["sales"],
     "tag_list": []
   }
```

### Phase 2: Question Answering (Runtime Phase)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    QUESTION ANSWERING FLOW                                  │
└────────────────────────────────────────────────────────────────────────────┘

USER QUESTION: "What are the top 5 customers by total sales?"

1. API Endpoint (answerQuestion)
   │
   GET /answerQuestion?question="What are the top 5 customers by total sales?"
   │
   ▼
2. Question Categorization (sdk_ai_tools.categorize_query)
   │
   LangChain Chain:
   │
   Prompt: "Categorize this question: {question}
           Context: Available databases and schemas
           Categories: SQL_CATEGORY, METADATA_CATEGORY, etc."
   │
   ├─► UniformLLM.invoke(prompt)
   ├─► LLM Response: "SQL_CATEGORY"
   │
   ▼
3. Vector Search for Relevant Tables (UniformVectorStore.search)
   │
   Step 3a: Embed the question
   │
   question = "What are the top 5 customers by total sales?"
   │
   UniformEmbeddings.embed_query(question)
   │
   query_vector = [0.234, -0.567, 0.890, ...]
   │
   Step 3b: Semantic search in vector database
   │
   ChromaDB.query({
     query_embeddings: [query_vector],
     n_results: 5,
     where: {"database_name": "sales"}  # Optional filter
   })
   │
   Results (sorted by cosine similarity):
   │
   [
     {
       id: "123",
       document: "Table: sales.customer...",
       metadata: {view_json: {...}, database_name: "sales"},
       distance: 0.15
     },
     {
       id: "456",
       document: "Table: sales.orders...",
       metadata: {view_json: {...}, database_name: "sales"},
       distance: 0.23
     }
   ]
   │
   ▼
4. SQL Generation (sdk_ai_tools.query_to_vql)
   │
   LangChain Chain with Retrieved Context:
   │
   Prompt Template:
   "Generate VQL query for: {question}

   Available Tables:
   {retrieved_table_schemas}

   Sample Data:
   {sample_data}

   VQL Syntax Rules:
   {vql_rules}

   Requirements:
   - Use proper VQL syntax
   - Include all necessary joins
   - Add appropriate filters
   - Limit results appropriately"
   │
   ├─► UniformLLM.invoke(prompt)
   │
   LLM Response:
   │
   VQL Query:
   "SELECT c.name as customer_name,
           SUM(o.total_amount) as total_sales
    FROM sales.customer c
    JOIN sales.orders o ON c.customer_id = o.customer_id
    GROUP BY c.name
    ORDER BY total_sales DESC
    LIMIT 5"
   │
   Query Explanation:
   "This query joins customers with orders, calculates total sales
    per customer, and returns top 5 by sales amount."
   │
   ▼
5. Query Validation & Fixing (sdk_ai_tools.query_fixer)
   │
   LangChain Chain:
   │
   Prompt: "Review and fix this VQL query if needed:
           Query: {vql_query}
           Question: {question}
           Common issues: syntax errors, missing joins, etc."
   │
   ├─► UniformLLM.invoke(prompt)
   ├─► Corrected VQL if needed
   │
   ▼
6. Query Execution (execute_vql)
   │
   POST to Data Catalog:
   │
   POST /public/api/askaquestion/execute
   {
     "vql": "SELECT c.name as customer_name, ...",
     "limit": 100
   }
   │
   ▼
7. Denodo Data Catalog Execution
   │
   ├─► Parse VQL
   ├─► Execute against Denodo VDP
   ├─► Return results
   │
   Response:
   {
     "rows": [
       {
         "values": [
           {"column": "customer_name", "value": "Acme Corp"},
           {"column": "total_sales", "value": 150000}
         ]
       },
       {
         "values": [
           {"column": "customer_name", "value": "Global Inc"},
           {"column": "total_sales", "value": 145000}
         ]
       },
       ...
     ]
   }
   │
   ▼
8. Error Handling & Retry Logic
   │
   If status_code == 500 (Error):
   │
   ├─► Extract error message
   ├─► Call query_fixer again with error context
   ├─► Retry execution (max 2 attempts)
   │
   If status_code == 499 (Empty result):
   │
   └─► Return empty result with explanation
   │
   ▼
9. Natural Language Answer Generation (sdk_ai_tools.answer_view)
   │
   LangChain Chain:
   │
   Prompt: "Generate a natural language answer:

           Question: {question}

           Query Results:
           {execution_results}

           Requirements:
           - Use markdown formatting
           - Create a summary table
           - Provide insights
           - Be concise and accurate"
   │
   ├─► UniformLLM.invoke(prompt)
   │
   LLM Response:
   │
   "Based on the data, here are the top 5 customers by total sales:

   | Customer Name | Total Sales |
   |--------------|-------------|
   | Acme Corp    | $150,000    |
   | Global Inc   | $145,000    |
   | Tech Solutions| $132,000   |
   | Retail Plus  | $128,000    |
   | Services Co  | $125,000    |

   Acme Corp leads with $150,000 in total sales, followed closely by
   Global Inc at $145,000. The top 5 customers account for $680,000
   in combined sales."
   │
   ▼
10. Generate Related Questions (sdk_ai_tools.related_questions)
    │
    Prompt: "Generate 3 related follow-up questions"
    │
    ├─► UniformLLM.invoke(prompt)
    │
    [
      "What is the average order value for these top customers?",
      "How has customer sales changed over time?",
      "Which products do these top customers buy most?"
    ]
    │
    ▼
11. Optional: Generate Visualization (if plot=true)
    │
    ├─► Determine chart type (bar, line, pie)
    ├─► Generate Python matplotlib code
    ├─► Execute code to create chart
    └─► Return base64 encoded image
    │
    ▼
12. Response to User
    │
    {
      "answer": "Based on the data, here are...",
      "sql_query": "SELECT c.name...",
      "query_explanation": "This query joins...",
      "execution_result": {
        "Row 1": [{"columnName": "customer_name", "value": "Acme Corp"}],
        ...
      },
      "related_questions": [...],
      "tables_used": ["sales.customer", "sales.orders"],
      "raw_graph": "base64_encoded_image",
      "tokens": {
        "input_tokens": 2341,
        "output_tokens": 456,
        "total_tokens": 2797
      },
      "sql_execution_time": 0.234,
      "vector_store_search_time": 0.045,
      "llm_time": 2.156,
      "total_execution_time": 2.435
    }
```

---

## DeepQuery Architecture

**DeepQuery** is an advanced analytical feature that uses "thinking" LLMs (like OpenAI o1/o3, Claude Sonnet with extended thinking) to perform multi-step deep analysis.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DEEPQUERY ARCHITECTURE                              │
└────────────────────────────────────────────────────────────────────────────┘

USER QUESTION: "Analyze sales trends and identify key growth opportunities"

1. DeepQuery Endpoint
   │
   POST /deepQuery
   {
     "question": "Analyze sales trends and identify key growth opportunities",
     "rows": 50,
     "execution_model": "thinking"
   }
   │
   ▼
2. Planning Phase (Thinking LLM)
   │
   Uses: OpenAI o3-mini / Claude Sonnet (with thinking enabled)
   │
   LangChain Agent with Tools:
   │
   ├─► Tool 1: search_views(query) - Find relevant tables
   ├─► Tool 2: get_view_schema(view_id) - Get detailed schema
   ├─► Tool 3: describe_question() - Break down the question
   │
   Thinking Process (Internal Chain of Thought):
   │
   "Let me think about this systematically:

   1. To analyze sales trends, I need:
      - Historical sales data (time-series)
      - Product categories
      - Geographic regions
      - Customer segments

   2. For growth opportunities, I should look at:
      - Underperforming segments
      - High-growth products
      - Seasonal patterns
      - Market penetration rates

   3. I'll need multiple queries:
      - Overall trends by time period
      - Sales by product category
      - Regional performance
      - Customer segment analysis

   4. Then I'll synthesize findings to identify opportunities"
   │
   Output: Analysis Plan
   │
   {
     "analysis_steps": [
       {
         "step": 1,
         "description": "Get overall sales trends by month",
         "required_views": ["sales.orders", "sales.order_items"],
         "expected_insights": "Identify growth/decline patterns"
       },
       {
         "step": 2,
         "description": "Analyze sales by product category",
         "required_views": ["sales.products", "sales.order_items"],
         "expected_insights": "Find best/worst performing categories"
       },
       {
         "step": 3,
         "description": "Regional performance analysis",
         "required_views": ["sales.customers", "sales.orders"],
         "expected_insights": "Identify high-growth regions"
       }
     ],
     "synthesis_approach": "Compare trends across dimensions to find opportunities"
   }
   │
   ▼
3. Execution Phase (Loop over Analysis Steps)
   │
   For each step in analysis_plan:
   │
   Step 3a: Generate Query
   │
   Uses: Thinking LLM or Regular LLM (configurable)
   │
   ├─► Vector search for relevant schemas
   ├─► Generate VQL query for this step
   │
   Example Query 1:
   "SELECT DATE_TRUNC('month', order_date) as month,
           SUM(total_amount) as monthly_sales,
           COUNT(DISTINCT order_id) as order_count
    FROM sales.orders
    WHERE order_date >= ADD_YEAR(NOW(), -2)
    GROUP BY month
    ORDER BY month"
   │
   Step 3b: Execute Query
   │
   ├─► Call Data Catalog execute API
   ├─► Retrieve results
   │
   Results 1: [
     {"month": "2023-01", "monthly_sales": 125000, "order_count": 450},
     {"month": "2023-02", "monthly_sales": 132000, "order_count": 475},
     ...
   ]
   │
   Step 3c: Intermediate Analysis
   │
   ├─► Feed results back to Thinking LLM
   ├─► Get insights for this step
   │
   Insights 1: "Sales show 15% YoY growth with strong Q4 performance"
   │
   Step 3d: Store Context
   │
   accumulated_context = {
     "step_1_results": results_1,
     "step_1_insights": insights_1
   }
   │
   [Repeat for all steps...]
   │
   ▼
4. Reporting Phase (Thinking LLM)
   │
   Input: All accumulated results and insights
   │
   LangChain Agent with Reporting Tools:
   │
   ├─► Tool 1: generate_visualization() - Create charts
   ├─► Tool 2: calculate_metrics() - Compute KPIs
   ├─► Tool 3: format_report() - Structure findings
   │
   Thinking Process:
   │
   "Now I need to synthesize all findings:

   From Step 1: 15% YoY growth, strong Q4
   From Step 2: Electronics +25%, Apparel -5%
   From Step 3: APAC +40%, EMEA +10%, Americas +5%

   Key Opportunities:
   1. Expand electronics in all regions (high growth)
   2. Improve apparel strategy (declining)
   3. Invest more in APAC (highest growth rate)
   4. Capitalize on Q4 momentum year-round"
   │
   Output: Comprehensive Report
   │
   {
     "executive_summary": "...",
     "detailed_findings": [...],
     "visualizations": [...],
     "recommendations": [...],
     "next_steps": [...]
   }
   │
   ▼
5. Response to User
   │
   {
     "report_id": "deepquery_123",
     "status": "completed",
     "analysis_plan": {...},
     "execution_results": [...],
     "final_report": {
       "summary": "Comprehensive markdown report",
       "charts": ["base64_chart1", "base64_chart2"],
       "insights": [...]
     },
     "execution_time": 45.2,
     "queries_executed": 5,
     "tokens_used": 25000
   }
```

**DeepQuery Agent Architecture**:

```
┌────────────────────────────────────────────────────────────────┐
│                    DeepQuery Agent System                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           Planning Agent (Thinking LLM)                  │ │
│  │  • Question decomposition                                │ │
│  │  • View discovery                                        │ │
│  │  • Analysis strategy                                     │ │
│  │  • Tool: search_views()                                  │ │
│  │  • Tool: get_view_schema()                               │ │
│  │  • Tool: describe_question()                             │ │
│  └───────────────────┬──────────────────────────────────────┘ │
│                      │ Analysis Plan                           │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Analysis Agent (Thinking/Regular LLM)            │ │
│  │  • Query generation                                      │ │
│  │  • Query execution                                       │ │
│  │  • Result interpretation                                 │ │
│  │  • Context accumulation                                  │ │
│  │  • Tool: generate_query()                                │ │
│  │  • Tool: execute_query()                                 │ │
│  │  • Tool: analyze_results()                               │ │
│  │  • Max loops: configurable (default: 10)                 │ │
│  └───────────────────┬──────────────────────────────────────┘ │
│                      │ Accumulated Results + Insights          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           Reporting Agent (Thinking LLM)                 │ │
│  │  • Synthesis of findings                                 │ │
│  │  • Visualization generation                              │ │
│  │  • Recommendation formulation                            │ │
│  │  • Report formatting                                     │ │
│  │  • Tool: generate_visualization()                        │ │
│  │  • Tool: calculate_metrics()                             │ │
│  │  • Tool: format_report()                                 │ │
│  │  • Max loops: configurable (default: 5)                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### LangChain Integration

The SDK uses **LangChain** extensively for:

1. **Prompt Templates**:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL generator for Denodo VQL..."),
    ("human", "{question}\n\nAvailable tables:\n{tables}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": q, "tables": schemas})
```

2. **Chain Composition**:
```python
# Multi-step chain
categorization_chain = prompt1 | llm | parser1
sql_generation_chain = prompt2 | llm | parser2
answer_chain = prompt3 | llm | parser3

# Sequential execution
category = categorization_chain.invoke({"question": q})
sql = sql_generation_chain.invoke({"question": q, "category": category})
answer = answer_chain.invoke({"sql": sql, "results": results})
```

3. **Streaming Responses**:
```python
async for chunk in llm.astream(prompt):
    yield chunk.content
```

4. **Token Tracking**:
```python
with get_usage_metadata_callback() as cb:
    response = llm.invoke(prompt)
    tokens = {
        'input_tokens': cb.input_tokens,
        'output_tokens': cb.output_tokens,
        'total_tokens': cb.total_tokens
    }
```

### State Management

**File**: `api/utils/state_manager.py`

The SDK uses an in-memory cache to reuse expensive resources:

```python
class StateManager:
    _llm_cache = {}
    _vector_store_cache = {}
    _embeddings_cache = {}

    def get_llm(provider, model, temperature, max_tokens):
        key = f"{provider}_{model}_{temperature}_{max_tokens}"
        if key not in _llm_cache:
            _llm_cache[key] = UniformLLM(...)
        return _llm_cache[key]

    def get_vector_store(provider, embeddings_provider, ...):
        # Similar caching for vector stores

    def get_embeddings(provider, model):
        # Similar caching for embeddings
```

### Error Handling & Retry Logic

The SDK implements sophisticated error handling:

```python
async def attempt_query_execution(vql_query, request, auth, ...):
    max_attempts = 2
    attempt = 0

    while attempt < max_attempts:
        status_code, execution_result = await execute_vql(
            vql_query, auth, request.vql_execute_rows_limit
        )

        if status_code == 200:
            return vql_query, execution_result, status_code

        elif status_code == 500:  # Execution error
            # Extract error message
            error_msg = execution_result

            # Use LLM to fix the query
            fixed_query = await query_fixer(
                question=request.question,
                query=vql_query,
                error_message=error_msg,
                llm=llm
            )

            vql_query = fixed_query
            attempt += 1

        elif status_code == 499:  # Empty result
            return vql_query, "No data found", status_code

    return vql_query, execution_result, status_code
```

### Authentication & Security

```python
from fastapi.security import HTTPBasic, HTTPBearer

security_basic = HTTPBasic()
security_bearer = HTTPBearer()

def authenticate(
    basic_credentials: HTTPBasicCredentials = Depends(security_basic),
    bearer_credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
):
    if bearer_credentials:
        return bearer_credentials.credentials  # OAuth token
    elif basic_credentials:
        return (basic_credentials.username, basic_credentials.password)
    else:
        raise HTTPException(status_code=401, detail="Authentication required")
```

### Incremental Metadata Updates

```python
def get_views_metadata_documents(..., last_update_timestamp_ms, incremental):
    if incremental and last_update_timestamp_ms:
        # Only fetch views modified since last sync
        data["updatedSince"] = last_update_timestamp_ms

    response = requests.post(DATA_CATALOG_METADATA_URL, json=data)
    metadata = response.json()

    # Get deleted views
    delete_view_ids = metadata.get('deletedViewIdentifiers', [])

    # Get detagged views (for tag-based sync)
    detagged_view_ids = metadata.get('detaggedViewIdentifiers', [])

    return processed_views, delete_view_ids, detagged_view_ids
```

### Performance Optimization

1. **Parallel Embedding Generation**:
```python
def add_views(views, parallel=True):
    if parallel:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.embeddings.embed_documents, [view])
                for view in views
            ]
            results = [f.result() for f in futures]
    else:
        results = [self.embeddings.embed_documents([view]) for view in views]
```

2. **Rate Limiting**:
```python
if self.rate_limit_rpm and self.rate_limit_rpm > 0:
    delay = 60.0 / self.rate_limit_rpm
    time.sleep(delay)
```

3. **Pagination**:
```python
# Fetch metadata in chunks
limit = 50
offset = 0

while True:
    data = {"offset": offset, "limit": limit}
    response = requests.post(url, json=data)
    views = response.json()['viewsDetails']

    if len(views) < limit:
        break

    offset += limit
```

---

## Configuration

**Key Environment Variables**:

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# Thinking LLM (for DeepQuery)
THINKING_LLM_PROVIDER=openai
THINKING_LLM_MODEL=o3-mini
THINKING_LLM_TEMPERATURE=1.0
THINKING_LLM_MAX_TOKENS=10240

# Embeddings
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-large

# Vector Store
VECTOR_STORE=chroma

# Data Catalog
DATA_CATALOG_URL=https://denodo-datacatalog.com/denodo-data-catalog/
DATA_CATALOG_VERIFY_SSL=1

# API Configuration
AI_SDK_HOST=0.0.0.0
AI_SDK_PORT=8008
AI_SDK_WORKERS=1
```

---

## Summary

The Denodo AI SDK implements a sophisticated **RAG (Retrieval-Augmented Generation)** architecture that:

1. **Ingests** database metadata from Denodo Platform into a vector database
2. **Searches** semantically for relevant tables when a user asks a question
3. **Generates** SQL (VQL) queries using LLMs with retrieved context
4. **Executes** queries against Denodo Platform
5. **Synthesizes** natural language answers from results
6. **Supports** advanced analytics through DeepQuery with thinking models

**Key Strengths**:
- Unified abstractions for LLMs, embeddings, and vector stores
- Comprehensive error handling and retry logic
- Incremental metadata synchronization
- Token tracking and performance metrics
- Support for 15+ LLM providers and 5+ vector stores
- Advanced multi-step reasoning with DeepQuery

This architecture enables enterprise users to query their data using natural language while maintaining security, accuracy, and scalability.
