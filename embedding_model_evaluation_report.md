# Banking Database Embedding Model Evaluation Report

## Evaluation Summary

This evaluation used 5 queries across different categories:

- Simple: 3 queries
- Medium: 2 queries

## Overall Results

| Model | Precision | Recall | F1 Score | MRR | Success Rate | Latency (s) |
|-------|-----------|--------|----------|-----|--------------|-------------|
| nomic-embed-text | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.1090 |
| mxbai-embed-large | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.1647 |
| bge-m3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.9348 |

## Results by Query Category

### nomic-embed-text

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Simple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| Medium | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |

### mxbai-embed-large

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Simple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| Medium | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |

### bge-m3

| Category | Precision | Recall | F1 Score | MRR | Queries |
|----------|-----------|--------|----------|-----|--------|
| Simple | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| Medium | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |


## Query Latency

| Model | Average (s) | Minimum (s) | Maximum (s) |
|-------|-------------|-------------|-------------|
| nomic-embed-text | 2.1090 | 2.0684 | 2.1516 |
| mxbai-embed-large | 2.1647 | 2.1029 | 2.2269 |
| bge-m3 | 2.9348 | 2.7949 | 3.0828 |

## Recommendations

- Best for precision: **nomic-embed-text** (0.0000)
- Best for recall: **nomic-embed-text** (0.0000)
- Best overall performance (F1): **nomic-embed-text** (0.0000)
- Fastest response time: **nomic-embed-text** (2.1090s)

### Recommended model(s) for production use:

- **nomic-embed-text**: Best overall retrieval performance. Best response time. Highest precision. Highest recall. 
