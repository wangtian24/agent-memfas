"""
MiniMax summarization for context compaction.

Condenses medium-relevance chunks to save tokens while preserving key information.
"""

from typing import List, Optional
import json
import httpx


class MiniMaxSummarizer:
    """
    Summarizes context chunks using MiniMax API.
    
    Uses MiniMax's Anthropic-compatible API to condense chunks
    while preserving key information.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.chat",
        model: str = "MiniMax-M2.1",
        target_tokens: int = 500,
    ):
        """
        Initialize summarizer.
        
        Args:
            api_key: MiniMax API key
            base_url: MiniMax API base URL
            model: Model to use for summarization
            target_tokens: Target token count for summaries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.target_tokens = target_tokens
        
    def summarize_chunk(
        self,
        chunk: str,
        prompt: str = "",
        preserve_keywords: List[str] = None,
    ) -> str:
        """
        Summarize a single chunk.
        
        Args:
            chunk: Content to summarize
            prompt: Current user prompt (for context)
            preserve_keywords: Keywords to ensure are preserved
            
        Returns:
            Summarized content
        """
        # Skip if chunk is already short
        if len(chunk.split()) < 50:
            return chunk
            
        keywords_hint = ""
        if preserve_keywords:
            keywords_hint = f" Preserve these keywords: {', '.join(preserve_keywords)}."
        
        system_prompt = f"""You are a context compression assistant. 
Your job is to summarize context chunks while preserving all key information 
relevant to the user's current task.

Target length: approximately {self.target_tokens} tokens.

Preserve:
- Key facts and decisions
- Technical details and code
- Important names, dates, numbers
- Any information relevant to: {prompt or 'general context'}

Remove:
- Filler words and redundant phrasing
- Repetitive explanations
- Irrelevant tangents

Output only the summarized content, no explanations.{keywords_hint}"""

        user_content = f"""Context chunk to summarize:

{chunk}

Current user focus: {prompt or 'general conversation'}

Summarize this chunk:"""

        try:
            response = self._call_api(system_prompt, user_content)
            return response.strip() if response else chunk
        except Exception as e:
            print(f"MiniMax summarization error: {e}")
            return chunk  # Return original on error
    
    def summarize_batch(
        self,
        chunks: List[str],
        prompt: str = "",
        preserve_keywords: List[str] = None,
    ) -> List[str]:
        """
        Summarize multiple chunks.
        
        Args:
            chunks: List of chunks to summarize
            prompt: Current user prompt
            preserve_keywords: Keywords to preserve
            
        Returns:
            List of summarized chunks
        """
        if not chunks:
            return []
            
        summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk, prompt, preserve_keywords)
            summaries.append(summary)
        return summaries
    
    def _call_api(self, system_prompt: str, user_content: str) -> str:
        """Make API call to MiniMax."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.target_tokens + 100,
            "temperature": 0.3,  # Low temp for consistent summaries
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            return ""
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4


def create_summarizer(
    api_key: str = None,
    config_path: str = None,
) -> MiniMaxSummarizer:
    """
    Factory function to create summarizer.
    
    Args:
        api_key: API key, or load from config
        config_path: Path to config file
        
    Returns:
        MiniMaxSummarizer instance
    """
    # Load from config if not provided
    if not api_key and config_path:
        import yaml
        from pathlib import Path
        
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            api_key = config.get("minimax_api_key") or config.get("api_key")
    
    # Get from environment
    if not api_key:
        api_key = __import__("os").getenv("MINIMAX_API_KEY", "")
    
    return MiniMaxSummarizer(api_key=api_key)
