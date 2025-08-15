from typing import List, Dict, Any, Optional
from ..models.content_insight import ContentChunk

#### chunking strategy section ##############################################

class ContentChunker:
    """Handles intelligent chunking of long-form content with context preservation"""
    
    def __init__(
        self, 
        chunk_size: int = 2000,  # tokens per chunk
        overlap_size: int = 200,  # overlap between chunks
        min_chunk_size: int = 500  # minimum viable chunk size
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_transcript(
        self, 
        transcript_data: Dict[str, Any],
        preserve_timestamps: bool = True
    ) -> List[ContentChunk]:
        """
        Chunk transcript with timestamp preservation
        
        Args:
            transcript_data: Whisper output with segments and words
            preserve_timestamps: Whether to maintain word-level timing
        """
        chunks = []
        
        if 'segments' not in transcript_data:
            # fallback for simple text
            return self._chunk_plain_text(transcript_data.get('text', ''))
        
        segments = transcript_data['segments']
        current_chunk_text = ""
        current_word_timestamps = []
        chunk_start_time = None
        chunk_id = 0
        
        for segment in segments:
            segment_text = segment.get('text', '')
            segment_start = segment.get('start', 0)
            segment_words = segment.get('words', [])
            
            # start new chunk if current would exceed size
            if (len(current_chunk_text) + len(segment_text) > self.chunk_size and 
                len(current_chunk_text) > self.min_chunk_size):
                
                # create chunk with overlap consideration
                chunk = self._create_chunk(
                    chunk_id, current_chunk_text, current_word_timestamps,
                    chunk_start_time, segment_start
                )
                chunks.append(chunk)
                
                # prepare next chunk with overlap
                current_chunk_text = self._get_overlap_text(current_chunk_text)
                current_word_timestamps = self._get_overlap_timestamps(current_word_timestamps)
                chunk_id += 1
            
            # set start time for first segment in chunk
            if chunk_start_time is None:
                chunk_start_time = segment_start
            
            # add segment to current chunk
            current_chunk_text += segment_text
            if preserve_timestamps and segment_words:
                current_word_timestamps.extend(segment_words)
        
        # add final chunk if it has content
        if current_chunk_text.strip():
            final_chunk = self._create_chunk(
                chunk_id, current_chunk_text, current_word_timestamps,
                chunk_start_time, segments[-1].get('end', chunk_start_time)
            )
            chunks.append(final_chunk)
        
        return chunks
    
    def _create_chunk(
        self, 
        chunk_id: int, 
        text: str, 
        word_timestamps: List[Dict], 
        start_time: float, 
        end_time: float
    ) -> ContentChunk:
        """Create a ContentChunk object"""
        return ContentChunk(
            chunk_id=f"chunk_{chunk_id:03d}",
            start_time=start_time,
            end_time=end_time,
            text=text.strip(),
            word_timestamps=word_timestamps
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from end of current chunk"""
        words = text.split()
        if len(words) <= self.overlap_size // 10:  # rough word estimate
            return text
        return ' '.join(words[-(self.overlap_size // 10):])
    
    def _get_overlap_timestamps(self, timestamps: List[Dict]) -> List[Dict]:
        """Extract overlap timestamps from end of current chunk"""
        overlap_count = min(len(timestamps), self.overlap_size // 10)
        return timestamps[-overlap_count:] if overlap_count > 0 else []
    
    def _chunk_plain_text(self, text: str) -> List[ContentChunk]:
        """Fallback chunking for plain text without timestamps"""
        chunks = []
        words = text.split()
        chunk_id = 0
        
        for i in range(0, len(words), self.chunk_size - self.overlap_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = ContentChunk(
                chunk_id=f"chunk_{chunk_id:03d}",
                text=chunk_text,
                word_timestamps=[]
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks

#### semantic chunking section ##############################################

class SemanticChunker(ContentChunker):
    """Enhanced chunker that considers semantic boundaries"""
    
    def chunk_transcript(self, transcript_data: Dict[str, Any], **kwargs) -> List[ContentChunk]:
        """Chunk with preference for sentence and speaker boundaries"""
        base_chunks = super().chunk_transcript(transcript_data, **kwargs)
        
        # refine chunks to break on sentence boundaries where possible
        refined_chunks = []
        for chunk in base_chunks:
            refined = self._refine_chunk_boundaries(chunk)
            refined_chunks.extend(refined)
        
        return refined_chunks
    
    def _refine_chunk_boundaries(self, chunk: ContentChunk) -> List[ContentChunk]:
        """Refine chunk to break on natural boundaries"""
        # for now, return as-is, but could implement sentence detection
        return [chunk]