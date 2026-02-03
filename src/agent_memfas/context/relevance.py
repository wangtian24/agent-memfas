"""
Relevance scoring for context chunks.
"""

from typing import Optional
import re


class RelevanceScorer:
    """
    Scores context chunks for relevance to current prompt.
    
    Formula:
        relevance_score = (memfas_similarity × memfas_weight) +
                         (recency_hours × recency_bonus) +
                         (importance_flag × importance_bonus)
    """
    
    def __init__(
        self,
        config,
        memfas_client: Optional[object] = None,
    ):
        """
        Initialize scorer.
        
        Args:
            config: ContextConfig instance
            memfas_client: Optional memfas client for similarity scoring
        """
        self.config = config
        self.memfas_client = memfas_client
        
    def score(
        self,
        chunk: str,
        prompt: str,
        recency_hours: float = 0.0,
        is_important: bool = False,
    ) -> float:
        """
        Score a chunk for relevance to prompt.
        
        Args:
            chunk: Context chunk content
            prompt: Current user prompt
            recency_hours: Hours since chunk was added
            is_important: Whether chunk is marked important
            
        Returns:
            Relevance score (0.0 to 1.0+)
        """
        # memfas similarity (if available)
        memfas_score = 0.0
        if self.memfas_client:
            try:
                results = self.memfas_client.search(prompt, limit=1)
                if results:
                    memfas_score = results[0].score if hasattr(results[0], 'score') else 0.5
            except Exception:
                pass
        else:
            # Fallback: simple keyword overlap
            memfas_score = self._keyword_overlap(chunk, prompt)
            
        # Compute final score
        score = (memfas_score * self.config.memfas_weight)
        
        # Add recency bonus
        score += recency_hours * self.config.recency_bonus
        
        # Add importance bonus
        if is_important:
            score += self.config.importance_bonus
            
        return score
    
    def _keyword_overlap(self, chunk: str, prompt: str) -> float:
        """
        Simple keyword overlap as fallback similarity metric.
        
        Returns:
            Overlap score (0.0 to 1.0)
        """
        # Tokenize (simple whitespace + punctuation)
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        
        if not prompt_words or not chunk_words:
            return 0.0
            
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'just', 'also', 'very', 'more',
        }
        
        chunk_keywords = chunk_words - stopwords
        prompt_keywords = prompt_words - stopwords
        
        if not prompt_keywords:
            return 0.0
            
        overlap = len(chunk_keywords & prompt_keywords)
        return min(overlap / len(prompt_keywords), 1.0)
    
    def classify(
        self,
        chunk: str,
        prompt: str,
        recency_hours: float = 0.0,
        is_important: bool = False,
    ) -> str:
        """
        Classify a chunk: KEEP, DROP, or SUMMARIZE.
        
        Args:
            chunk: Context chunk content
            prompt: Current user prompt
            recency_hours: Hours since chunk was added
            is_important: Whether chunk is marked important
            
        Returns:
            Classification: "keep", "drop", or "summarize"
        """
        score = self.score(chunk, prompt, recency_hours, is_important)
        
        if score >= self.config.relevance_keep_threshold:
            return "keep"
        elif score <= self.config.relevance_cutoff:
            return "drop"
        else:
            return "summarize"
    
    def score_batch(
        self,
        chunks: list[str],
        prompt: str,
        recency_hours: list[float] = None,
        is_important: list[bool] = None,
    ) -> list[float]:
        """
        Score multiple chunks.
        
        Args:
            chunks: List of context chunks
            prompt: Current user prompt
            recency_hours: Optional list of hours since each chunk
            is_important: Optional list of importance flags
            
        Returns:
            List of relevance scores
        """
        recency_hours = recency_hours or [0.0] * len(chunks)
        is_important = is_important or [False] * len(chunks)
        
        return [
            self.score(c, p, r, i)
            for c, p, r, i in zip(chunks, prompt, recency_hours, is_important)
        ]
