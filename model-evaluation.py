import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time

# Langchain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configuration constants
CHROMA_DB_PATH = "./chroma_db"  # Local path to store the Chroma DB
OLLAMA_BASE_URL = "http://localhost:11434"  # Update with your Ollama API URL

# Embedding model configurations
EMBEDDING_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-m3"
]

def load_vector_stores():
    """
    Load the existing Chroma vector stores using Langchain.
    
    Returns:
        dict: Dictionary of Chroma vector store objects
    """
    vector_stores = {}
    
    for model_name in EMBEDDING_MODELS:
        try:
            # Create the embedding function for this model
            embedding = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=model_name
            )
            
            # Set collection name and persist directory
            collection_name = f"denodo_metadata_{model_name.replace('-', '_')}"
            persist_directory = os.path.join(CHROMA_DB_PATH, collection_name)
            
            # Load the existing vector store
            if os.path.exists(persist_directory):
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embedding,
                    persist_directory=persist_directory
                )
                
                # Get the collection count to verify it loaded correctly
                collection_count = vectorstore._collection.count()
                
                vector_stores[model_name] = vectorstore
                print(f"Loaded collection: {collection_name} with {collection_count} documents")
            else:
                print(f"Collection directory not found for {model_name}. Please run the loader script first.")
        
        except Exception as e:
            print(f"Error loading vector store for {model_name}: {e}")
    
    return vector_stores

def create_banking_eval_queries():
    """
    Create evaluation queries based on the banking schema with specific SQL categories.
    
    Returns:
        list: List of evaluation queries with relevance info
    """
    eval_queries = []
    
    # Simple Queries (Simple SELECT statements)
    simple_queries = [
        {
            "query": "Show me the first name, last name and phone number of all the customers",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["first_name", "last_name", "phone_number"],
            "sql_category": "Simple",
            "explanation": "Basic SELECT of specific columns"
        },
        {
            "query": "Show me the first 10 customers we have",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["customer_id", "first_name", "last_name"],
            "sql_category": "Simple",
            "explanation": "SELECT with LIMIT"
        },
        {
            "query": "How many customers are named John?",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["first_name"],
            "sql_category": "Simple",
            "explanation": "Simple COUNT with WHERE condition"
        }
    ]
    eval_queries.extend(simple_queries)
    
    # Medium Queries (With Conditions)
    medium_queries = [
        {
            "query": "List the customers that are in California",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["first_name", "last_name", "state"],
            "sql_category": "Medium",
            "explanation": "SELECT with WHERE condition on state"
        },
        {
            "query": "How many customers have checking account?",
            "relevant_tables": ["bv_bank_acct", "bv_bank_customers"],
            "relevant_columns": ["acct_type", "customer_id"],
            "sql_category": "Medium",
            "explanation": "COUNT with JOIN and WHERE condition"
        },
        {
            "query": "Can you retrieve the ID, customer information, and loan amount for the loans that have an amount greater than 400000?",
            "relevant_tables": ["bv_bank_loans", "bv_bank_customers"],
            "relevant_columns": ["loan_id", "customer_id", "loan_amount"],
            "sql_category": "Medium",
            "explanation": "SELECT with JOIN and numeric comparison"
        },
        {
            "query": "Could you list the information of the customer named John that lives in California?",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["first_name", "state"],
            "sql_category": "Medium",
            "explanation": "SELECT with multiple WHERE conditions"
        },
        {
            "query": "Could you show me loans opened 38 days ago?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with date calculation"
        },
        {
            "query": "Could you show me loans opened on April 20th, 2023?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with specific date"
        },
        {
            "query": "Could you show me loans opened on 11/8/2022?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with specific date in different format"
        },
        {
            "query": "Could you show me loans that are opened in the last 50 days?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with date range"
        },
        {
            "query": "Could you show me loans that opened last month?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with relative date range"
        },
        {
            "query": "What are the loans created in Q1 2022?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with quarter filtering"
        },
        {
            "query": "How many loans do we have in the year 2021?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "COUNT with year filtering"
        },
        {
            "query": "What are the loans in March 2021?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["date_created"],
            "sql_category": "Medium",
            "explanation": "SELECT with month and year filtering"
        },
        {
            "query": "List all the customers in alphabetical order by last name",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["last_name"],
            "sql_category": "Medium",
            "explanation": "SELECT with ORDER BY"
        },
        {
            "query": "Please provide a list of customers ordered by their loan amounts in descending order",
            "relevant_tables": ["bv_bank_customers", "bv_bank_loans"],
            "relevant_columns": ["customer_id", "loan_amount"],
            "sql_category": "Medium",
            "explanation": "SELECT with JOIN and ORDER BY DESC"
        },
        {
            "query": "Please provide a list of customers ordered by their loan amounts in descending order. If there are same amounts order by their last name",
            "relevant_tables": ["bv_bank_customers", "bv_bank_loans"],
            "relevant_columns": ["customer_id", "loan_amount", "last_name"],
            "sql_category": "Medium",
            "explanation": "SELECT with JOIN and multiple ORDER BY conditions"
        }
    ]
    eval_queries.extend(medium_queries)
    
    # Complex Queries (Aggregations and GroupBy)
    complex_queries = [
        {
            "query": "Can you retrieve the ID, customer information, and loan amount for the loans that have an amount greater than the average loan amount among all the users?",
            "relevant_tables": ["bv_bank_loans", "bv_bank_customers"],
            "relevant_columns": ["loan_id", "customer_id", "loan_amount"],
            "sql_category": "Complex",
            "explanation": "SELECT with subquery for AVG comparison"
        },
        {
            "query": "Could you show me the list of loan officers and the loans that they are in charge of?",
            "relevant_tables": ["bv_bank_loanofficers", "bv_bank_loans"],
            "relevant_columns": ["loan_officer_id", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with INNER JOIN"
        },
        {
            "query": "Could you show me the list of loan officers and the loans that they are in charge of? Also include the officers who are not assigned to any loans yet.",
            "relevant_tables": ["bv_bank_loanofficers", "bv_bank_loans"],
            "relevant_columns": ["loan_officer_id", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with LEFT JOIN"
        },
        {
            "query": "What is the average number of loans that each loan officer is in charge of?",
            "relevant_tables": ["bv_bank_loanofficers", "bv_bank_loans"],
            "relevant_columns": ["loan_officer_id", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with JOIN, GROUP BY and AVG"
        },
        {
            "query": "Show me the number of customers by region",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": ["state", "city"],
            "sql_category": "Complex",
            "explanation": "SELECT with GROUP BY and COUNT"
        },
        {
            "query": "Show me the total number of loans by property",
            "relevant_tables": ["bv_bank_loans", "bv_bank_properties"],
            "relevant_columns": ["property_id", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with JOIN, GROUP BY and COUNT"
        },
        {
            "query": "Show me the total loans by month and state",
            "relevant_tables": ["bv_bank_loans", "bv_bank_customers", "bv_bank_properties"],
            "relevant_columns": ["date_created", "state", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with multiple JOINs, GROUP BY multiple fields"
        },
        {
            "query": "What is the average amount of loan across all users?",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": ["loan_amount"],
            "sql_category": "Complex",
            "explanation": "SELECT with AVG aggregation"
        },
        {
            "query": "What is the total loan amount for the loans we have opened in CA?",
            "relevant_tables": ["bv_bank_loans", "bv_bank_properties"],
            "relevant_columns": ["loan_amount", "state"],
            "sql_category": "Complex",
            "explanation": "SELECT with JOIN, WHERE and SUM"
        },
        {
            "query": "Which customers have more than one loan with us?",
            "relevant_tables": ["bv_bank_customers", "bv_bank_loans"],
            "relevant_columns": ["customer_id", "loan_id"],
            "sql_category": "Complex",
            "explanation": "SELECT with GROUP BY and HAVING"
        }
    ]
    eval_queries.extend(complex_queries)
    
    # Additional categories matching the chart in the image
    
    # Direct Questions
    direct_queries = [
        {
            "query": "What columns are in the customers table?",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": [],
            "sql_category": "Direct",
            "explanation": "Schema exploration - specific table"
        },
        {
            "query": "What foreign keys connect loans to customers?",
            "relevant_tables": ["bv_bank_loans", "bv_bank_customers"],
            "relevant_columns": ["customer_id"],
            "sql_category": "Direct",
            "explanation": "Schema exploration - relationships"
        },
        {
            "query": "Tell me about the properties table schema",
            "relevant_tables": ["bv_bank_properties"],
            "relevant_columns": [],
            "sql_category": "Direct",
            "explanation": "Schema exploration - table details"
        }
    ]
    eval_queries.extend(direct_queries)
    
    # Implied Questions
    implied_queries = [
        {
            "query": "Find customers who might be at risk of default",
            "relevant_tables": ["bv_bank_customers", "bv_bank_loans", "bv_bank_underwriting"],
            "relevant_columns": ["credit_score", "financial_history", "payment_id"],
            "sql_category": "Implied",
            "explanation": "Implied concept - risk assessment"
        },
        {
            "query": "Which properties are likely undervalued?",
            "relevant_tables": ["bv_bank_properties"],
            "relevant_columns": ["property_value", "city", "state"],
            "sql_category": "Implied", 
            "explanation": "Implied concept - value assessment"
        },
        {
            "query": "Who are our best customers?",
            "relevant_tables": ["bv_bank_customers", "bv_bank_loans", "bv_bank_acct"],
            "relevant_columns": ["balance", "loan_amount"],
            "sql_category": "Implied",
            "explanation": "Implied concept - customer value"
        }
    ]
    eval_queries.extend(implied_queries)
    
    # Unclear Questions
    unclear_queries = [
        {
            "query": "Show me the loan data",
            "relevant_tables": ["bv_bank_loans"],
            "relevant_columns": [],
            "sql_category": "Unclear",
            "explanation": "Ambiguous - which loan attributes?"
        },
        {
            "query": "How are our properties doing?",
            "relevant_tables": ["bv_bank_properties", "bv_bank_loans"],
            "relevant_columns": ["property_value", "loan_amount"],
            "sql_category": "Unclear",
            "explanation": "Ambiguous - performance metric undefined"
        },
        {
            "query": "Find important customer information",
            "relevant_tables": ["bv_bank_customers"],
            "relevant_columns": [],
            "sql_category": "Unclear", 
            "explanation": "Ambiguous - 'important' is subjective"
        }
    ]
    eval_queries.extend(unclear_queries)
    
    return eval_queries

def prepare_relevance_data(queries):
    """
    Prepare relevance data for evaluating retrieval performance.
    
    Args:
        queries (list): List of query dictionaries
        
    Returns:
        dict: Dictionary mapping queries to relevant content identifiers
    """
    query_relevance = {}
    
    for query_obj in queries:
        query = query_obj["query"]
        relevant_tables = query_obj["relevant_tables"]
        relevant_columns = query_obj["relevant_columns"]
        
        # Create full table.column combinations for relevance checking
        relevant_content = []
        for table in relevant_tables:
            # Add the table itself
            relevant_content.append(table.lower())
            
            # Add table.column combinations
            for column in relevant_columns:
                relevant_content.append(f"{table.lower()}.{column.lower()}")
        
        query_relevance[query] = {
            "relevant_content": relevant_content,
            "category": query_obj["sql_category"]
        }
    
    return query_relevance

def check_content_relevance(doc_content, relevant_items):
    """
    Check if a document's content is relevant based on tables/columns.
    
    Args:
        doc_content (str): Document content text
        relevant_items (list): List of relevant table/column identifiers
        
    Returns:
        tuple: (is_relevant, matched_items)
    """
    doc_content_lower = doc_content.lower()
    matched_items = []
    
    for item in relevant_items:
        if item in doc_content_lower:
            matched_items.append(item)
    
    return len(matched_items) > 0, matched_items

def run_evaluation(vector_stores, queries, query_relevance, k=5):
    """
    Run comprehensive evaluation on all models using the prepared queries.
    
    Args:
        vector_stores (dict): Dictionary of vector stores
        queries (list): List of query dictionaries
        query_relevance (dict): Dictionary mapping queries to relevant items
        k (int): Number of results to retrieve
        
    Returns:
        dict: Comprehensive evaluation results
    """
    results = {}
    
    for model_name, vectorstore in vector_stores.items():
        print(f"\nEvaluating model: {model_name}")
        model_results = []
        
        for query_obj in tqdm(queries, desc=f"Queries"):
            query_text = query_obj["query"]
            category = query_obj["sql_category"]
            relevant_content = query_relevance[query_text]["relevant_content"]
            
            try:
                # Get retrievals from vector store with k=5
                retrievals = vectorstore.similarity_search_with_score(
                    query=query_text,
                    k=k
                )
                
                # Evaluate relevance of each retrieved document
                relevant_hits = []
                retrievals_info = []
                
                for i, (doc, score) in enumerate(retrievals):
                    # Check document relevance
                    is_relevant, matched_items = check_content_relevance(
                        doc.page_content, 
                        relevant_content
                    )
                    
                    # Store retrieval info
                    doc_info = {
                        "position": i,
                        "score": score,
                        "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "is_relevant": is_relevant,
                        "matched_items": matched_items
                    }
                    retrievals_info.append(doc_info)
                    
                    if is_relevant:
                        relevant_hits.append(doc_info)
                
                # Calculate metrics
                precision = len(relevant_hits) / len(retrievals) if retrievals else 0
                recall = len(set(item for hit in relevant_hits for item in hit["matched_items"])) / len(relevant_content) if relevant_content else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # Calculate MRR (Mean Reciprocal Rank)
                mrr = 0
                for i, (doc, _) in enumerate(retrievals):
                    is_relevant, _ = check_content_relevance(doc.page_content, relevant_content)
                    if is_relevant:
                        mrr = 1.0 / (i + 1)
                        break
                
                # Store result for this query
                result = {
                    "query": query_text,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mrr": mrr,
                    "retrieved_count": len(retrievals),
                    "relevant_count": len(relevant_content),
                    "relevant_hits_count": len(relevant_hits),
                    "retrievals": retrievals_info,
                    "relevant_hits": relevant_hits
                }
                model_results.append(result)
                
            except Exception as e:
                print(f"Error evaluating query '{query_text}': {e}")
                model_results.append({
                    "query": query_text,
                    "category": category,
                    "error": str(e),
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "mrr": 0
                })
        
        results[model_name] = model_results
    
    return results

def measure_query_latency(vector_stores, sample_queries, k=5, runs=3):
    """
    Measure query latency for all models.
    
    Args:
        vector_stores (dict): Dictionary of vector stores
        sample_queries (list): List of queries to test
        k (int): Number of results to retrieve
        runs (int): Number of runs per query
        
    Returns:
        dict: Latency metrics by model
    """
    latency_metrics = {}
    
    for model_name, vectorstore in vector_stores.items():
        print(f"\nMeasuring latency for {model_name}...")
        query_times = []
        
        for query in tqdm(sample_queries, desc="Queries"):
            for run in range(runs):
                try:
                    start_time = time.time()
                    vectorstore.similarity_search_with_score(query=query, k=k)
                    end_time = time.time()
                    query_times.append(end_time - start_time)
                except Exception as e:
                    print(f"Error during latency test for query '{query}': {e}")
        
        if query_times:
            latency_metrics[model_name] = {
                "avg_latency": sum(query_times) / len(query_times),
                "min_latency": min(query_times),
                "max_latency": max(query_times),
                "std_dev": np.std(query_times) if len(query_times) > 1 else 0,
                "samples": len(query_times)
            }
        else:
            print(f"No successful latency measurements for {model_name}")
            latency_metrics[model_name] = {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "std_dev": 0,
                "samples": 0
            }
    
    return latency_metrics

def calculate_metrics_by_category(results):
    """
    Calculate metrics aggregated by query category.
    
    Args:
        results (dict): Evaluation results
        
    Returns:
        dict: Metrics by category for each model
    """
    metrics_by_category = {}
    
    for model_name, model_results in results.items():
        category_metrics = {}
        
        # Group results by category
        for result in model_results:
            category = result.get("category", "Unknown")
            
            if category not in category_metrics:
                category_metrics[category] = {
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "mrr": [],
                    "query_count": 0
                }
            
            if "error" not in result:
                category_metrics[category]["precision"].append(result["precision"])
                category_metrics[category]["recall"].append(result["recall"])
                category_metrics[category]["f1"].append(result["f1"])
                category_metrics[category]["mrr"].append(result["mrr"])
                category_metrics[category]["query_count"] += 1
        
        # Calculate averages for each category
        for category, metrics in category_metrics.items():
            if metrics["query_count"] > 0:
                metrics["avg_precision"] = sum(metrics["precision"]) / metrics["query_count"]
                metrics["avg_recall"] = sum(metrics["recall"]) / metrics["query_count"]
                metrics["avg_f1"] = sum(metrics["f1"]) / metrics["query_count"]
                metrics["avg_mrr"] = sum(metrics["mrr"]) / metrics["query_count"]
        
        metrics_by_category[model_name] = category_metrics
    
    return metrics_by_category

def calculate_overall_metrics(results):
    """
    Calculate overall metrics across all queries for each model.
    
    Args:
        results (dict): Evaluation results
        
    Returns:
        dict: Overall metrics by model
    """
    overall_metrics = {}
    
    for model_name, model_results in results.items():
        if not model_results:
            print(f"No results for model {model_name}")
            continue
        
        precision_values = [r.get("precision", 0) for r in model_results if "error" not in r]
        recall_values = [r.get("recall", 0) for r in model_results if "error" not in r]
        f1_values = [r.get("f1", 0) for r in model_results if "error" not in r]
        mrr_values = [r.get("mrr", 0) for r in model_results if "error" not in r]
        
        if precision_values:
            overall_metrics[model_name] = {
                "avg_precision": sum(precision_values) / len(precision_values),
                "avg_recall": sum(recall_values) / len(recall_values),
                "avg_f1": sum(f1_values) / len(f1_values),
                "avg_mrr": sum(mrr_values) / len(mrr_values),
                "query_count": len(precision_values),
                "success_rate": sum(1 for v in precision_values if v > 0) / len(precision_values)
            }
        else:
            overall_metrics[model_name] = {
                "avg_precision": 0,
                "avg_recall": 0,
                "avg_f1": 0,
                "avg_mrr": 0,
                "query_count": 0,
                "success_rate": 0
            }
    
    return overall_metrics

def visualize_results(overall_metrics, metrics_by_category, latency_metrics=None):
    """
    Create visualizations of the evaluation results.
    
    Args:
        overall_metrics (dict): Overall metrics by model
        metrics_by_category (dict): Metrics by category for each model
        latency_metrics (dict, optional): Latency metrics by model
    """
    models = list(overall_metrics.keys())
    
    # 1. Overall Metrics Comparison
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Overall Model Performance', fontsize=16)
    
    # Precision
    axs[0, 0].bar(models, [overall_metrics[m]["avg_precision"] for m in models])
    axs[0, 0].set_title("Average Precision")
    axs[0, 0].set_ylim(0, 1)
    
    # Recall
    axs[0, 1].bar(models, [overall_metrics[m]["avg_recall"] for m in models])
    axs[0, 1].set_title("Average Recall")
    axs[0, 1].set_ylim(0, 1)
    
    # F1
    axs[1, 0].bar(models, [overall_metrics[m]["avg_f1"] for m in models])
    axs[1, 0].set_title("Average F1 Score")
    axs[1, 0].set_ylim(0, 1)
    
    # MRR
    axs[1, 1].bar(models, [overall_metrics[m]["avg_mrr"] for m in models])
    axs[1, 1].set_title("Mean Reciprocal Rank")
    axs[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("overall_performance.png")
    plt.close()
    
    # 2. Performance by Category
    categories = set()
    for model, cat_metrics in metrics_by_category.items():
        categories.update(cat_metrics.keys())
    categories = sorted(list(categories))
    
    for metric in ["avg_precision", "avg_recall", "avg_f1", "avg_mrr"]:
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(categories))
        width = 0.2
        offset = -width * (len(models) - 1) / 2
        
        for i, model in enumerate(models):
            values = []
            for category in categories:
                if category in metrics_by_category[model] and metric in metrics_by_category[model][category]:
                    values.append(metrics_by_category[model][category][metric])
                else:
                    values.append(0)
            
            plt.bar(x + offset + i * width, values, width, label=model)
        
        plt.xlabel('Category')
        plt.ylabel(metric.replace('avg_', '').capitalize())
        plt.title(f'{metric.replace("avg_", "").capitalize()} by Query Category')
        plt.xticks(x, categories)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{metric}_by_category.png")
        plt.close()
    
    # 3. Latency Comparison
    if latency_metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot average latency
        plt.bar(models, [latency_metrics[m]["avg_latency"] for m in models])
        
        # Add error bars for std dev
        plt.errorbar(
            models, 
            [latency_metrics[m]["avg_latency"] for m in models],
            yerr=[latency_metrics[m]["std_dev"] for m in models],
            fmt='o', color='black'
        )
        
        plt.ylabel('Seconds')
        plt.title('Average Query Latency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add min/max annotations
        for i, model in enumerate(models):
            plt.annotate(
                f"Min: {latency_metrics[model]['min_latency']:.3f}s\nMax: {latency_metrics[model]['max_latency']:.3f}s",
                xy=(i, latency_metrics[model]["avg_latency"]),
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
            )
        
        plt.tight_layout()
        plt.savefig("latency_comparison.png")
        plt.close()
    
    print("Visualizations saved to disk.")

def generate_evaluation_report(overall_metrics, metrics_by_category, latency_metrics, queries):
    """
    Generate a comprehensive evaluation report in Markdown format.
    
    Args:
        overall_metrics (dict): Overall metrics by model
        metrics_by_category (dict): Metrics by category for each model
        latency_metrics (dict): Latency metrics by model
        queries (list): List of query dictionaries
        
    Returns:
        str: Markdown formatted report
    """
    models = list(overall_metrics.keys())
    categories = sorted(set(q["sql_category"] for q in queries))
    
    report = "# Banking Schema RAG Evaluation Report\n\n"
    
    # Introduction and summary
    report += "## Overview\n\n"
    report += f"This report evaluates {len(models)} embedding models on their ability to retrieve relevant information "
    report += f"for SQL query generation across {len(categories)} question categories. "
    report += f"A total of {len(queries)} queries were used in the evaluation.\n\n"
    
    # Count queries by category
    report += "### Query Categories\n\n"
    query_counts = {}
    for query in queries:
        category = query["sql_category"]
        query_counts[category] = query_counts.get(category, 0) + 1
    
    report += "| Category | Count | Description |\n"
    report += "|----------|-------|-------------|\n"
    
    category_descriptions = {
        "Simple": "Basic SELECT queries with no conditions or joins",
        "Medium": "Queries with WHERE clauses, filtering and sorting",
        "Complex": "Queries with JOINs, GROUP BY, and aggregations",
        "Direct": "Questions directly about database schema",
        "Implied": "Questions requiring inference about concepts not directly in schema",
        "Unclear": "Ambiguous questions requiring interpretation"
    }
    
    for category in sorted(query_counts.keys()):
        desc = category_descriptions.get(category, "")
        report += f"| {category} | {query_counts[category]} | {desc} |\n"
    
    # Overall results section
    report += "\n## Overall Results\n\n"
    report += "| Model | Precision | Recall | F1 Score | MRR | Success Rate | Avg Latency (s) |\n"
    report += "|-------|-----------|--------|----------|-----|--------------|----------------|\n"
    
    for model in models:
        report += f"| {model} | {overall_metrics[model]['avg_precision']:.4f} | "
        report += f"{overall_metrics[model]['avg_recall']:.4f} | "
        report += f"{overall_metrics[model]['avg_f1']:.4f} | "
        report += f"{overall_metrics[model]['avg_mrr']:.4f} | "
        report += f"{overall_metrics[model]['success_rate']:.4f} | "
        
        if model in latency_metrics:
            report += f"{latency_metrics[model]['avg_latency']:.4f} |\n"
        else:
            report += "N/A |\n"
    
    # Best model analysis
    report += "\n### Best Performing Models\n\n"
    
    best_precision = max(models, key=lambda m: overall_metrics[m]['avg_precision'])
    best_recall = max(models, key=lambda m: overall_metrics[m]['avg_recall'])
    best_f1 = max(models, key=lambda m: overall_metrics[m]['avg_f1'])
    best_mrr = max(models, key=lambda m: overall_metrics[m]['avg_mrr'])
    fastest = min(models, key=lambda m: latency_metrics[m]['avg_latency'] if m in latency_metrics and latency_metrics[m]['avg_latency'] > 0 else float('inf'))
    
    report += f"- **Best for Precision**: {best_precision} ({overall_metrics[best_precision]['avg_precision']:.4f})\n"
    report += f"- **Best for Recall**: {best_recall} ({overall_metrics[best_recall]['avg_recall']:.4f})\n"
    report += f"- **Best Overall Performance (F1)**: {best_f1} ({overall_metrics[best_f1]['avg_f1']:.4f})\n"
    report += f"- **Best for Relevant First Result (MRR)**: {best_mrr} ({overall_metrics[best_mrr]['avg_mrr']:.4f})\n"
    report += f"- **Fastest Response Time**: {fastest} ({latency_metrics[fastest]['avg_latency']:.4f}s)\n\n"
    
    # Category breakdown section
    report += "\n## Performance by Query Category\n\n"
    
    for model in models:
        report += f"### {model}\n\n"
        report += "| Category | Precision | Recall | F1 Score | MRR | Queries |\n"
        report += "|----------|-----------|--------|----------|-----|--------|\n"
        
        for category in categories:
            if (category in metrics_by_category[model] and 
                "avg_precision" in metrics_by_category[model][category]):
                cat_metrics = metrics_by_category[model][category]
                report += f"| {category} | {cat_metrics['avg_precision']:.4f} | "
                report += f"{cat_metrics['avg_recall']:.4f} | "
                report += f"{cat_metrics['avg_f1']:.4f} | "
                report += f"{cat_metrics['avg_mrr']:.4f} | "
                report += f"{cat_metrics['query_count']} |\n"
            else:
                report += f"| {category} | N/A | N/A | N/A | N/A | 0 |\n"
        
        report += "\n"
    
    # Best model per category section
    report += "\n## Best Model by Category\n\n"
    report += "| Category | Best Model | F1 Score | Precision | Recall |\n"
    report += "|----------|------------|----------|-----------|--------|\n"
    
    for category in categories:
        best_model = None
        best_f1 = -1
        best_precision = -1
        best_recall = -1
        
        for model in models:
            if (category in metrics_by_category[model] and 
                "avg_f1" in metrics_by_category[model][category] and 
                metrics_by_category[model][category]["avg_f1"] > best_f1):
                best_model = model
                best_f1 = metrics_by_category[model][category]["avg_f1"]
                best_precision = metrics_by_category[model][category]["avg_precision"]
                best_recall = metrics_by_category[model][category]["avg_recall"]
        
        if best_model:
            report += f"| {category} | {best_model} | {best_f1:.4f} | {best_precision:.4f} | {best_recall:.4f} |\n"
        else:
            report += f"| {category} | N/A | N/A | N/A | N/A |\n"
    
    # Latency analysis
    report += "\n## Query Latency Analysis\n\n"
    report += "| Model | Avg Latency (s) | Min (s) | Max (s) | Std Dev |\n"
    report += "|-------|-----------------|---------|---------|--------|\n"
    
    for model in models:
        if model in latency_metrics:
            report += f"| {model} | {latency_metrics[model]['avg_latency']:.4f} | "
            report += f"{latency_metrics[model]['min_latency']:.4f} | "
            report += f"{latency_metrics[model]['max_latency']:.4f} | "
            report += f"{latency_metrics[model]['std_dev']:.4f} |\n"
        else:
            report += f"| {model} | N/A | N/A | N/A | N/A |\n"
    
    # Final recommendation
    report += "\n## Recommendations\n\n"
    
    # Calculate weighted score (70% F1, 30% latency)
    weighted_scores = {}
    for model in models:
        f1_score = overall_metrics[model]['avg_f1']
        if model in latency_metrics and latency_metrics[model]['avg_latency'] > 0:
            # Normalize latency score (lower is better)
            max_latency = max(latency_metrics[m]['avg_latency'] for m in models if m in latency_metrics)
            latency_score = 1 - (latency_metrics[model]['avg_latency'] / max_latency)
            weighted_scores[model] = 0.7 * f1_score + 0.3 * latency_score
        else:
            weighted_scores[model] = f1_score
    
    best_overall = max(models, key=lambda m: weighted_scores[m])
    
    report += f"Based on a weighted combination of retrieval accuracy (70%) and query latency (30%), "
    report += f"the recommended model for the banking schema RAG application is:\n\n"
    report += f"**{best_overall}**\n\n"
    
    # Model strengths and weaknesses
    report += "### Model Strengths\n\n"
    
    for model in models:
        report += f"**{model}**:\n"
        
        # Find categories where this model performs best
        best_categories = []
        for category in categories:
            if category in metrics_by_category[model] and "avg_f1" in metrics_by_category[model][category]:
                if metrics_by_category[model][category]["avg_f1"] > 0.6:  # Threshold for "good" performance
                    best_categories.append((category, metrics_by_category[model][category]["avg_f1"]))
        
        # Sort by F1 score
        best_categories.sort(key=lambda x: x[1], reverse=True)
        
        if best_categories:
            report += "- Strong performance in categories:\n"
            for category, score in best_categories[:3]:  # Top 3 categories
                report += f"  - {category} (F1: {score:.4f})\n"
        else:
            report += "- No standout performance categories\n"
        
        # Latency comment
        if model in latency_metrics:
            avg_latency = latency_metrics[model]['avg_latency']
            all_latencies = [latency_metrics[m]['avg_latency'] for m in models if m in latency_metrics]
            all_latencies.sort()
            
            if avg_latency == all_latencies[0]:
                report += f"- Fastest query response time ({avg_latency:.4f}s)\n"
            elif avg_latency <= all_latencies[len(all_latencies)//2]:
                report += f"- Above average query speed ({avg_latency:.4f}s)\n"
            else:
                report += f"- Below average query speed ({avg_latency:.4f}s)\n"
        
        report += "\n"
    
    # Save report to file
    with open("banking_schema_evaluation_report.md", "w") as f:
        f.write(report)
    
    print("Report saved to banking_schema_evaluation_report.md")
    
    return report

def analyze_example_queries(results, num_examples=3):
    """
    Analyze and display example query results for each category.
    
    Args:
        results (dict): Evaluation results
        num_examples (int): Number of examples to show per category
    """
    # Get a reference model
    model = next(iter(results.keys()))
    model_results = results[model]
    
    # Group by category
    by_category = {}
    for result in model_results:
        category = result.get("category", "Unknown")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)
    
    print("\n===== EXAMPLE QUERY ANALYSIS =====")
    
    for category, category_results in by_category.items():
        print(f"\n== {category} Queries ==\n")
        
        # Sort by F1 score
        category_results.sort(key=lambda x: x.get("f1", 0), reverse=True)
        
        # Display top examples
        for i, result in enumerate(category_results[:num_examples]):
            query = result.get("query", "Unknown query")
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            print(f"Query {i+1}: {query}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Show top relevant hits
            relevant_hits = result.get("relevant_hits", [])
            if relevant_hits:
                print("  Top relevant retrievals:")
                for j, hit in enumerate(sorted(relevant_hits, key=lambda x: x.get("position", 99))[:2]):
                    position = hit.get("position", -1)
                    print(f"    [{position+1}] Matches: {', '.join(hit.get('matched_items', []))}")
            else:
                print("  No relevant retrievals")
            
            print()

if __name__ == "__main__":
    print("Banking Schema RAG Evaluation System")
    print("=" * 50)
    
    # Load vector stores
    print("\nLoading vector stores...")
    vector_stores = load_vector_stores()
    
    if not vector_stores:
        print("No vector stores found. Please run the loader script first.")
        exit(1)
    
    # Create evaluation queries
    print("\nPreparing banking evaluation queries...")
    eval_queries = create_banking_eval_queries()
    query_relevance = prepare_relevance_data(eval_queries)
    
    print(f"Created {len(eval_queries)} evaluation queries across multiple SQL categories:")
    for category in sorted(set(q["sql_category"] for q in eval_queries)):
        count = sum(1 for q in eval_queries if q["sql_category"] == category)
        print(f"- {category}: {count} queries")
    
    # Show example queries
    print("\nSample queries by category:")
    for category in sorted(set(q["sql_category"] for q in eval_queries)):
        category_queries = [q for q in eval_queries if q["sql_category"] == category]
        if category_queries:
            print(f"\n{category} Queries:")
            for i, q in enumerate(category_queries[:2]):  # Show first 2 of each category
                print(f"  {i+1}. {q['query']}")
                print(f"     Relevant tables: {', '.join(q['relevant_tables'])}")
                if q['relevant_columns']:
                    print(f"     Relevant columns: {', '.join(q['relevant_columns'])}")
    
    # Prepare latency test queries
    latency_test_queries = [
        "Show customer information",
        "List loans by amount",
        "Find properties in California",
        "Show accounts with balance over 10000",
        "Calculate average loan amount",
        "Show loan officers and their loans",
        "Find customers with multiple accounts",
        "Count loans by state"
    ]
    
    # Ask for confirmation before running the evaluation
    print("\nBefore running the evaluation, please ensure that:")
    print("1. Ollama is running and accessible at the specified URL")
    print("2. All required embedding models are installed in Ollama")
    print(f"3. The k value for retrieval is set to 5")
    
    run_eval = input("\nRun evaluation now? (yes/no): ")
    if run_eval.lower() != "yes":
        print("Exiting without running evaluation. Edit the script and run again when ready.")
        exit(0)
    
    # Run the evaluation
    print("\nRunning evaluation...")
    results = run_evaluation(vector_stores, eval_queries, query_relevance, k=5)
    
    # Measure latency
    print("\nMeasuring query latency...")
    latency_metrics = measure_query_latency(vector_stores, latency_test_queries, k=5)
    
    # Calculate metrics
    metrics_by_category = calculate_metrics_by_category(results)
    overall_metrics = calculate_overall_metrics(results)
    
    # Print overall metrics
    print("\nOverall Metrics:")
    for model_name, metrics in overall_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Precision: {metrics['avg_precision']:.4f}")
        print(f"  Recall: {metrics['avg_recall']:.4f}")
        print(f"  F1 Score: {metrics['avg_f1']:.4f}")
        print(f"  MRR: {metrics['avg_mrr']:.4f}")
        print(f"  Success Rate: {metrics['success_rate']:.4f}")
    
    # Print latency metrics
    print("\nLatency Metrics:")
    for model_name, metrics in latency_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Avg Latency: {metrics['avg_latency']:.4f} seconds")
        print(f"  Min Latency: {metrics['min_latency']:.4f} seconds")
        print(f"  Max Latency: {metrics['max_latency']:.4f} seconds")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(overall_metrics, metrics_by_category, latency_metrics)
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    generate_evaluation_report(overall_metrics, metrics_by_category, latency_metrics, eval_queries)
    
    # Analyze example queries
    analyze_example_queries(results)
    
    print("\nEvaluation complete!")
    print("Generated files:")
    print("- banking_schema_evaluation_report.md - Comprehensive evaluation report")
    print("- overall_performance.png - Overall performance comparison")
    print("- avg_precision_by_category.png - Precision across categories")
    print("- avg_recall_by_category.png - Recall across categories")
    print("- avg_f1_by_category.png - F1 score across categories")
    print("- avg_mrr_by_category.png - MRR across categories")
    print("- latency_comparison.png - Latency comparison")