# Banking Schema RAG Evaluation System

A comprehensive system for evaluating and utilizing Retrieval-Augmented Generation (RAG) for SQL query generation against a banking database schema.

## Overview

This project provides tools to:

1. Evaluate multiple embedding models for sample data - bank schema in Denodo platform via Denodo AI SDK
2. Generate SQL queries from natural language questions using the best embedding model
3. Categorize queries by complexity (Simple, Medium, Complex)
4. Provide detailed performance analytics and visualizations

## Components

The project consists of two main components:

### 1. Evaluation System (`banking-schema-evaluation.py`)

Evaluates embedding models based on their ability to retrieve relevant schema information for SQL query generation.

- Supports multiple embedding models (nomic-embed-text, mxbai-embed-large, bge-m3)
- Tests against 40+ queries categorized by SQL complexity
- Generates comprehensive performance reports and visualizations
- Measures precision, recall, F1 score, MRR, and latency

### 2. SQL RAG System (`banking-rag-sql-system.py`)

Interactive system to generate SQL queries for a banking database using the best embedding model.

- Uses Ollama for embeddings and LLM
- Features specialized SQL prompting
- Categorizes questions by complexity
- Provides example queries by category
- Shows retrieved schema information

## System Requirements

- Python 3.8+
- Ollama running locally with required models:
  - Embedding models: nomic-embed-text, mxbai-embed-large, bge-m3
  - LLM model: mistral (or any model of your choice)

## Installation

1. Clone the repository:

Install the required packages:

bashCopypip install langchain langchain_community pandas matplotlib numpy tqdm

Start Ollama and ensure the required models are available:

bashCopy# Pull required models (if not already available)
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull bge-m3
ollama pull mistral
Usage
Running the Evaluation System
bashCopypython banking-schema-evaluation.py
This will:

Load vector stores for all embedding models
Run evaluation queries against banking schema
Generate performance metrics and visualizations
Create a comprehensive evaluation report

Running the SQL RAG System
bashCopypython banking-rag-sql-system.py
In the interactive mode:

Type your natural language question about the banking database
The system will categorize it and generate an appropriate SQL query
Type "examples" to see sample queries by category
Type "help" to see available commands
Type "exit" to quit

Banking Schema
The system is designed for a banking database with the following tables:

bv_bank_customers: Customer personal information
bv_bank_acct: Bank accounts information
bv_bank_loans: Loan details
bv_bank_loanofficers: Information about loan officers
bv_bank_payments: Loan payment records
bv_bank_properties: Property information for mortgage loans
bv_bank_underwriting: Loan risk assessment data
bv_bank_rates: Interest rate information
Plus several views combining these tables

Query Categories
The system evaluates and supports queries in these categories:
Simple
Basic SELECT statements:

"Show me the first name, last name and phone number of all customers"
"Show me the first 10 customers we have"
"How many customers are named John?"

Medium
Queries with WHERE clauses and conditions:

"List customers that are in California"
"How many customers have checking accounts?"
"Show loans opened in the last 50 days"
"List customers in alphabetical order by last name"

Complex
Queries with JOINs, GROUP BY, and aggregations:

"Show me average loan amount across all users"
"Find customers with more than one loan"
"Show loan officers and the loans they manage"
"Show number of customers by region"

Extending the System
Adding New Embedding Models

Update the EMBEDDING_MODELS list in both scripts
Ensure the models are available in Ollama
Re-run the evaluation to compare performance

Customizing Evaluation Queries
Modify the create_banking_eval_queries() function in the evaluation script to add or change queries.
