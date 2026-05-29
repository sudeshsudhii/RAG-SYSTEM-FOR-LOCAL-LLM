"""
Query classification module.

Routes queries into categories to optimize processing:
- Greeting / Small Talk: Handled directly without retrieval.
- RAG Query: Passed to the retrieval pipeline.
- Unsupported: Rejected gracefully.
"""

from typing import Literal
from src.utils import setup_logger
from src.prompts import CLASSIFICATION_PROMPT

logger = setup_logger(__name__)

QueryCategory = Literal["Greeting", "Small Talk", "RAG Query", "Unsupported"]

class QueryClassifier:
    """Classifies user queries using an LLM to determine the processing route."""

    def __init__(self, llm_provider):
        """
        Initialize with an LLM provider.
        
        Args:
            llm_provider: Instance of LLMProvider.
        """
        self.llm = llm_provider

    def classify(self, query: str) -> QueryCategory:
        """
        Classify the query into one of the known categories.
        
        Args:
            query: The user's input string.
            
        Returns:
            The determined QueryCategory.
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            # We don't want to use standard system prompt for classification
            response = self.llm.generate(prompt, system_prompt="You are a strict classifier. Respond ONLY with the category name.")
            result = response.strip()
            
            # Map loosely to exact categories
            if "Greeting" in result:
                return "Greeting"
            elif "Small Talk" in result:
                return "Small Talk"
            elif "Unsupported" in result:
                return "Unsupported"
            else:
                return "RAG Query"  # Default to RAG if uncertain
                
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to RAG Query.")
            return "RAG Query"
