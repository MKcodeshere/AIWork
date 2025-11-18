"""
Query Engine for Epstein Research Assistant
Handles querying Gemini with File Search and formatting responses
"""
from typing import Dict, List, Optional, Any
from google import genai
from google.genai import types


class QueryEngine:
    """Engine for querying documents and formatting responses"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        source_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize Query Engine

        Args:
            api_key: Gemini API key
            model_name: Model to use (gemini-2.5-flash or gemini-2.5-pro)
            source_mapping: Mapping of document IDs to source paths
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.source_mapping = source_mapping or {}

    def query(
        self,
        question: str,
        file_search_store,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the File Search store

        Args:
            question: User's question
            file_search_store: File Search Store object
            system_instruction: Optional system instruction for the model

        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not file_search_store:
            raise ValueError("File Search store not provided")

        try:
            # Default system instruction
            if not system_instruction:
                system_instruction = """You are an expert research assistant analyzing the Epstein Files dataset.

Your role:
- Provide accurate, well-sourced answers based on the documents
- Always cite your sources with specific references
- If information is not in the documents, clearly state that
- Present facts objectively without speculation
- When multiple documents discuss the same topic, synthesize the information
- Highlight any contradictions or discrepancies you find

Format your responses clearly with:
1. Direct answer to the question
2. Supporting evidence from documents
3. Source citations"""

            # Create the query with File Search tool
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[file_search_store.name]
                            )
                        )
                    ]
                )
            )

            # Extract answer
            answer = response.text if hasattr(response, 'text') else str(response)

            # Extract citations if available
            citations = self._extract_citations(response)

            # Format response
            return {
                'answer': answer,
                'citations': citations,
                'model': self.model_name,
                'success': True
            }

        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'citations': [],
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }

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
            # Check if response has grounding metadata
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata

                    # Extract grounding chunks
                    if hasattr(grounding, 'grounding_chunks'):
                        for chunk in grounding.grounding_chunks:
                            citation = {
                                'text': getattr(chunk, 'text', ''),
                                'source': getattr(chunk, 'source', ''),
                            }

                            # Try to get source path from mapping
                            if hasattr(chunk, 'metadata'):
                                metadata = chunk.metadata
                                if isinstance(metadata, dict):
                                    doc_id = metadata.get('doc_id', '')
                                    if doc_id in self.source_mapping:
                                        citation['source_path'] = self.source_mapping[doc_id]

                            citations.append(citation)

            # Also check for citation_metadata
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'citation_metadata'):
                    citation_metadata = candidate.citation_metadata

                    if hasattr(citation_metadata, 'citations'):
                        for cite in citation_metadata.citations:
                            citations.append({
                                'text': getattr(cite, 'text', ''),
                                'source': getattr(cite, 'source', ''),
                                'start_index': getattr(cite, 'start_index', None),
                                'end_index': getattr(cite, 'end_index', None)
                            })

        except Exception as e:
            print(f"Warning: Could not extract citations: {str(e)}")

        return citations

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
            for i, citation in enumerate(citations, 1):
                source_path = citation.get('source_path', citation.get('source', 'Unknown'))
                formatted += f"{i}. {source_path}\n"

                # Add excerpt if available
                if citation.get('text'):
                    excerpt = citation['text'][:200] + "..." if len(citation['text']) > 200 else citation['text']
                    formatted += f"   > {excerpt}\n"

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
