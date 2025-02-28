# Banking Schema RAG Evaluation Report

## Overview

This report evaluates 3 embedding models on their ability to retrieve relevant information for SQL query generation across 6 question categories. A total of 37 queries were used in the evaluation.

### Query Categories

| Category | Count | Description |
|----------|-------|-------------|
| Complex | 10 | Queries with JOINs, GROUP BY, and aggregations |
| Direct | 3 | Questions directly about database schema |
| Implied | 3 | Questions requiring inference about concepts not directly in schema |
| Medium | 15 | Queries with WHERE clauses, filtering and sorting |
| Simple | 3 | Basic SELECT queries with no conditions or joins |
| Unclear | 3 | Ambiguous questions requiring interpretation |

## Overall Results

| Model | Precision | Recall | F1 Score | MRR | Success Rate | Avg Latency (s) |
|-------|-----------|--------|----------|-----|--------------|----------------|
| nomic-embed-text | 0.6324 | 0.4613 | 0.4920 | 0.7995 | 1.0000 | 2.1087 |
| mxbai-embed-large | 0.6811 | 0.5495 | 0.5766 | 0.8874 | 1.0000 | 2.2463 |
| bge-m3 | 0.7892 | 0.5698 | 0.6295 | 0.9414 | 1.0000 | 3.1256 |

### Best Performing Models

- **Best for Precision**: bge-m3 (0.7892)
- **Best for Recall**: bge-m3 (0.5698)
- **Best Overall Performance (F1)**: bge-m3 (0.6295)
- **Best for Relevant First Result (MRR)**: bge-m3 (0.9414)
- **Fastest Response Time**: nomic-embed-text (2.1087s)


## Performance by Query Category

### nomic-embed-text

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Complex | 0.5600 | 0.4458 | 0.4596 | 0.7750 | 10 |
| Direct | 0.6667 | 0.7500 | 0.7037 | 0.8333 | 3 |
| Implied | 0.6667 | 0.1481 | 0.2179 | 0.8333 | 3 |
| Medium | 0.5867 | 0.4222 | 0.4634 | 0.7556 | 15 |
| Simple | 0.9333 | 0.4167 | 0.5714 | 1.0000 | 3 |
| Unclear | 0.7333 | 0.7778 | 0.7262 | 0.8333 | 3 |

### mxbai-embed-large

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Complex | 0.7600 | 0.6125 | 0.6551 | 0.8833 | 10 |
| Direct | 0.6000 | 0.7500 | 0.6264 | 0.6667 | 3 |
| Implied | 0.7333 | 0.2778 | 0.4015 | 1.0000 | 3 |
| Medium | 0.6267 | 0.5028 | 0.5466 | 0.8667 | 15 |
| Simple | 0.8000 | 0.4167 | 0.5372 | 1.0000 | 3 |
| Unclear | 0.6000 | 0.7778 | 0.6296 | 1.0000 | 3 |

### bge-m3

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Complex | 0.8000 | 0.6125 | 0.6795 | 0.9000 | 10 |
| Direct | 0.7333 | 1.0000 | 0.8201 | 1.0000 | 3 |
| Implied | 0.7333 | 0.2778 | 0.3962 | 1.0000 | 3 |
| Medium | 0.7867 | 0.5028 | 0.5957 | 0.9556 | 15 |
| Simple | 0.8667 | 0.4167 | 0.5543 | 1.0000 | 3 |
| Unclear | 0.8000 | 0.7778 | 0.7495 | 0.8333 | 3 |


## Best Model by Category

| Category | Best Model | F1 Score | Precision | Recall |
|----------|------------|----------|-----------|--------|
| Complex | bge-m3 | 0.6795 | 0.8000 | 0.6125 |
| Direct | bge-m3 | 0.8201 | 0.7333 | 1.0000 |
| Implied | mxbai-embed-large | 0.4015 | 0.7333 | 0.2778 |
| Medium | bge-m3 | 0.5957 | 0.7867 | 0.5028 |
| Simple | nomic-embed-text | 0.5714 | 0.9333 | 0.4167 |
| Unclear | bge-m3 | 0.7495 | 0.8000 | 0.7778 |

## Query Latency Analysis

| Model | Avg Latency (s) | Min (s) | Max (s) | Std Dev |
|-------|-----------------|---------|---------|--------|
| nomic-embed-text | 2.1087 | 2.0838 | 2.1635 | 0.0212 |
| mxbai-embed-large | 2.2463 | 2.1288 | 2.6294 | 0.1172 |
| bge-m3 | 3.1256 | 2.8433 | 3.7364 | 0.2788 |

## Recommendations

Based on a weighted combination of retrieval accuracy (70%) and query latency (30%), the recommended model for the banking schema RAG application is:

**mxbai-embed-large**

### Model Strengths

**nomic-embed-text**:
- Strong performance in categories:
  - Unclear (F1: 0.7262)
  - Direct (F1: 0.7037)
- Fastest query response time (2.1087s)

**mxbai-embed-large**:
- Strong performance in categories:
  - Complex (F1: 0.6551)
  - Unclear (F1: 0.6296)
  - Direct (F1: 0.6264)
- Above average query speed (2.2463s)

**bge-m3**:
- Strong performance in categories:
  - Direct (F1: 0.8201)
  - Unclear (F1: 0.7495)
  - Complex (F1: 0.6795)
- Below average query speed (3.1256s)

