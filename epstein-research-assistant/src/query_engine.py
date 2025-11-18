"""
Query Engine for Epstein Research Assistant
Uses Gemini File Search for retrieval + OpenAI for generation
"""
from typing import Dict, List, Optional, Any
from google import genai
from google.genai import types
from openai import OpenAI


class QueryEngine:
    """Engine for querying documents using hybrid RAG approach"""

    def __init__(
        self,
        gemini_api_key: str,
        openai_api_key: str,
        llm_model: str = "gpt-4o-mini",
        retrieval_model: str = "gemini-2.5-flash",
        source_mapping: Optional[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        Initialize Query Engine with hybrid approach

        Args:
            gemini_api_key: Gemini API key (for File Search retrieval)
            openai_api_key: OpenAI API key (for LLM generation)
            llm_model: OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)
            retrieval_model: Gemini model for File Search
            source_mapping: Mapping of document IDs to source paths
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.retrieval_model = retrieval_model
        self.source_mapping = source_mapping or {}
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(
        self,
        question: str,
        file_search_store,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query using Gemini File Search + OpenAI LLM

        Args:
            question: User's question
            file_search_store: Gemini File Search Store object
            system_instruction: Optional system instruction

        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not file_search_store:
            raise ValueError("File Search store not provided")

        try:
            # Step 1: Use Gemini File Search to retrieve relevant documents
            print(f"🔍 Retrieving relevant documents from Gemini File Search...")

            retrieval_response = self.gemini_client.models.generate_content(
                model=self.retrieval_model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[file_search_store.name]
                            )
                        )
                    ]
                )
            )

            # Step 2: Extract retrieved context
            retrieved_context = self._extract_context(retrieval_response)
            citations = self._extract_citations(retrieval_response)

            print(f"✅ Retrieved {len(citations)} relevant documents")

            # Step 3: Build prompt for OpenAI
            if not system_instruction:
                system_instruction = """You are an expert research assistant analyzing the Epstein Files dataset.

Your role:
- Provide accurate, well-sourced answers based on the retrieved documents
- Be objective and factual
- If information is not in the provided context, clearly state that
- Present facts without speculation
- Synthesize information from multiple documents when relevant

Format your responses clearly and professionally."""

            # Create context from retrieved documents
            context_text = self._format_context(retrieved_context, citations)

            user_prompt = f"""Based on the following retrieved documents, please answer the question.

QUESTION: {question}

RETRIEVED DOCUMENTS:
{context_text}

Please provide a clear, factual answer based only on the information in these documents. Include specific references to the documents when making claims."""

            # Step 4: Generate answer using OpenAI
            print(f"🤖 Generating answer with {self.llm_model}...")

            completion = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = completion.choices[0].message.content

            print(f"✅ Generated answer ({completion.usage.total_tokens} tokens)")

            # Return formatted response
            return {
                'answer': answer,
                'citations': citations,
                'retrieved_context': retrieved_context,
                'model': self.llm_model,
                'retrieval_model': self.retrieval_model,
                'tokens_used': completion.usage.total_tokens,
                'success': True
            }

        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'citations': [],
                'model': self.llm_model,
                'success': False,
                'error': str(e)
            }

    def _extract_context(self, response) -> List[str]:
        """
        Extract text chunks from Gemini response

        Args:
            response: Gemini API response

        Returns:
            List of retrieved text chunks
        """
        context = []

        try:
            # Extract from grounding metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata

                    if hasattr(grounding, 'grounding_chunks'):
                        for chunk in grounding.grounding_chunks:
                            if hasattr(chunk, 'text'):
                                context.append(chunk.text)

            # If no grounding metadata, try to extract from response text
            if not context and hasattr(response, 'text'):
                context.append(response.text)

        except Exception as e:
            print(f"Warning: Could not extract context: {str(e)}")

        return context

    def _extract_citations(self, response) -> List[Dict]:
        """
        Extract citations from Gemini response

        Args:
            response: Gemini API response

        Returns:
            List of citation dictionaries
        """
        citations = []

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Try grounding metadata
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata

                    if hasattr(grounding, 'grounding_chunks'):
                        for i, chunk in enumerate(grounding.grounding_chunks):
                            citation = {
                                'index': i + 1,
                                'text': getattr(chunk, 'text', '')[:300],  # Preview
                                'source': getattr(chunk, 'source', 'Unknown'),
                            }

                            # Try to get source path from mapping
                            if hasattr(chunk, 'metadata'):
                                metadata = chunk.metadata
                                if isinstance(metadata, dict):
                                    doc_id = metadata.get('doc_id', '')
                                    if doc_id in self.source_mapping:
                                        citation['source_path'] = self.source_mapping[doc_id]

                            citations.append(citation)

                # Also check citation_metadata
                if hasattr(candidate, 'citation_metadata'):
                    citation_metadata = candidate.citation_metadata

                    if hasattr(citation_metadata, 'citations'):
                        for cite in citation_metadata.citations:
                            citations.append({
                                'text': getattr(cite, 'text', '')[:300],
                                'source': getattr(cite, 'source', 'Unknown'),
                                'start_index': getattr(cite, 'start_index', None),
                                'end_index': getattr(cite, 'end_index', None)
                            })

        except Exception as e:
            print(f"Warning: Could not extract citations: {str(e)}")

        return citations

    def _format_context(self, context_chunks: List[str], citations: List[Dict]) -> str:
        """
        Format retrieved context for OpenAI prompt

        Args:
            context_chunks: List of retrieved text chunks
            citations: List of citation info

        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant documents were retrieved."

        formatted = []

        for i, chunk in enumerate(context_chunks):
            source = citations[i]['source'] if i < len(citations) else 'Unknown'
            formatted.append(f"--- Document {i+1} (Source: {source}) ---\n{chunk}\n")

        return "\n".join(formatted)

    def format_response_with_citations(self, result: Dict) -> str:
        """
        Format query result with citations

        Args:
            result: Query result dictionary

        Returns:
            Formatted response string
        """
        if not result.get('success'):
            return f"❌ {result.get('answer', 'Query failed')}"

        formatted = f"{result['answer']}\n\n"

        # Add citations if available
        citations = result.get('citations', [])
        if citations:
            formatted += "---\n### 📚 Sources:\n\n"
            for citation in citations:
                source_path = citation.get('source_path', citation.get('source', 'Unknown'))
                formatted += f"{citation.get('index', '?')}. **{source_path}**\n"

                # Add excerpt if available
                if citation.get('text'):
                    excerpt = citation['text']
                    formatted += f"   > {excerpt}...\n\n"

        # Add model info
        formatted += f"\n---\n*Generated by {result['model']} using Gemini File Search*"
        if result.get('tokens_used'):
            formatted += f" ({result['tokens_used']} tokens)"

        return formatted

    def batch_query(
        self,
        questions: List[str],
        file_search_store,
        system_instruction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries

        Args:
            questions: List of questions
            file_search_store: File Search Store object
            system_instruction: Optional system instruction

        Returns:
            List of query results
        """
        results = []

        for question in questions:
            result = self.query(
                question=question,
                file_search_store=file_search_store,
                system_instruction=system_instruction
            )
            results.append(result)

        return results

    def update_source_mapping(self, source_mapping: Dict[str, str]):
        """
        Update the source mapping

        Args:
            source_mapping: New source mapping dictionary
        """
        self.source_mapping = source_mapping
