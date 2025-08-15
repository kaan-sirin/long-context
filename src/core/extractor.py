import os
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

from ..models.content_insight import ContentInsight, TimestampedItem, ContentChunk
from ..models.content_insight import seconds_to_timestamp, create_youtube_link

#### extraction engine section ##############################################

class InsightExtractor:
    """Handles AI-powered insight extraction from content chunks"""
    
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_insights(
        self, 
        chunks: List[ContentChunk], 
        extraction_goal: str,
        source_info: Dict[str, Any]
    ) -> ContentInsight:
        """
        Extract insights from content chunks using iterative processing
        
        Args:
            chunks: List of content chunks to process
            extraction_goal: User-specified goal (e.g., "extract book recommendations")
            source_info: Source metadata
        """
        
        # process each chunk individually
        chunk_insights = []
        for chunk in chunks:
            chunk_insight = self._process_chunk(chunk, extraction_goal)
            chunk_insights.append(chunk_insight)
        
        # synthesize insights across chunks
        final_insights = self._synthesize_insights(
            chunk_insights, extraction_goal, source_info
        )
        
        return final_insights
    
    def _process_chunk(self, chunk: ContentChunk, extraction_goal: str) -> Dict[str, Any]:
        """Process individual chunk to extract relevant insights"""
        
        system_prompt = (
            "# Role\n"
            "You are an expert content analyst extracting insights from video transcripts.\n\n"
            "# Task\n"
            f"Extract insights related to: {extraction_goal}\n\n"
            "# Guidelines\n"
            "- Focus specifically on the extraction goal\n"
            "- Extract key insights, actionable items, and notable quotes\n"
            "- Be precise and specific\n"
            "- If no relevant content found, return empty lists\n\n"
            "# Output Format\n"
            "Return JSON with these fields:\n"
            "- key_insights: [list of important findings as strings]\n"
            "- action_items: [list of actionable recommendations as strings]\n"
            "- quotes: [list of notable direct quotes as strings]\n"
            "- relevance_score: float 0-1\n\n"
        )
        
        user_prompt = (
            f"# Extraction Goal\n"
            f"{extraction_goal}\n\n"
            f"# Content Chunk\n"
            f"Timestamp Range: {seconds_to_timestamp(chunk.start_time or 0)} - "
            f"{seconds_to_timestamp(chunk.end_time or 0)}\n\n"
            f"Content:\n{chunk.text}\n\n"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # using gpt-4o as gpt-5 might not be available
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            result['chunk_info'] = {
                'chunk_id': chunk.chunk_id,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'word_timestamps': chunk.word_timestamps
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return {
                'key_insights': [],
                'action_items': [],
                'quotes': [],
                'relevance_score': 0.0,
                'chunk_info': {'chunk_id': chunk.chunk_id}
            }
    
    def _synthesize_insights(
        self, 
        chunk_insights: List[Dict[str, Any]], 
        extraction_goal: str,
        source_info: Dict[str, Any]
    ) -> ContentInsight:
        """Synthesize insights from all chunks into final result"""
        
        all_insights = []
        all_actions = []
        all_quotes = []
        
        # collect and timestamp all insights
        for chunk_result in chunk_insights:
            chunk_info = chunk_result.get('chunk_info', {})
            
            # process key insights (handle both structured and plain string formats)
            for insight in chunk_result.get('key_insights', []):
                content = str(insight)  # just convert to string regardless of format
                timestamped_insight = self._create_timestamped_item_with_reference(
                    content, '', chunk_info, source_info
                )
                all_insights.append(timestamped_insight)
            
            # process action items
            for action in chunk_result.get('action_items', []):
                content = str(action)
                timestamped_action = self._create_timestamped_item_with_reference(
                    content, '', chunk_info, source_info
                )
                all_actions.append(timestamped_action)
            
            # process quotes
            for quote in chunk_result.get('quotes', []):
                content = str(quote)
                timestamped_quote = self._create_timestamped_item_with_reference(
                    content, '', chunk_info, source_info
                )
                all_quotes.append(timestamped_quote)
        
        # create final insight object
        from ..models.content_insight import SourceInfo
        source = SourceInfo(**source_info)
        
        return ContentInsight(
            extraction_goal=extraction_goal,
            source_info=source,
            key_insights=all_insights,
            action_items=all_actions,
            quotes=all_quotes,
            processing_chunks=len(chunk_insights)
        )
    
    def _create_timestamped_item_with_reference(
        self, 
        content: str,
        text_reference: str,
        chunk_info: Dict[str, Any], 
        source_info: Dict[str, Any]
    ) -> TimestampedItem:
        """Create timestamped item with precise timestamp based on text reference"""
        
        # get the chunk start time - this is the main issue
        chunk_start = chunk_info.get('start_time', 0)
        
        # for now, just use chunk start time since text reference matching isn't working
        start_time = chunk_start
        
        timestamp_display = seconds_to_timestamp(start_time)
        
        # create clickable URL for YouTube
        source_url = None
        if source_info.get('source_type') == 'youtube' and source_info.get('url'):
            source_url = create_youtube_link(source_info['url'], start_time)
        
        return TimestampedItem(
            content=content,
            timestamp_seconds=start_time,
            timestamp_display=timestamp_display,
            source_url=source_url
        )
    
    def _find_precise_timestamp(self, text_reference: str, chunk_info: Dict[str, Any]) -> float:
        """Find precise timestamp for a text reference using word-level timestamps"""
        chunk_start = chunk_info.get('start_time', 0)
        
        if not text_reference or not chunk_info.get('word_timestamps'):
            return chunk_start
        
        # find the first few words from the text reference
        ref_words = [word.lower().strip() for word in text_reference.split()[:3]]
        
        for word_data in chunk_info.get('word_timestamps', []):
            word_text = word_data.get('word', '').lower().strip()
            # remove punctuation for better matching
            clean_word = ''.join(c for c in word_text if c.isalnum())
            
            if clean_word and any(clean_word in ref_word or ref_word in clean_word for ref_word in ref_words):
                return word_data.get('start', chunk_start)
        
        # fallback to chunk start time
        return chunk_start
    
    def _create_timestamped_item(
        self, 
        content: str, 
        chunk_info: Dict[str, Any], 
        source_info: Dict[str, Any]
    ) -> TimestampedItem:
        """Create timestamped item with appropriate linking (legacy method)"""
        return self._create_timestamped_item_with_reference(
            content, '', chunk_info, source_info
        )