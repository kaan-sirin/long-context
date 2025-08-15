from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

#### content models section ##################################################

class TimestampedItem(BaseModel):
    """Base class for items with timestamp information"""
    content: str
    timestamp_seconds: Optional[float] = None  # for precise word-level timing
    timestamp_display: Optional[str] = None    # human-readable "12:34"
    source_url: Optional[str] = None          # clickable link for YouTube
    confidence: Optional[float] = None        # whisper confidence score

class SourceInfo(BaseModel):
    """Information about the source content"""
    source_type: str  # "youtube", "podcast", "text"
    title: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    processed_at: datetime

class ContentChunk(BaseModel):
    """Individual chunk of processed content"""
    chunk_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: str
    word_timestamps: List[Dict[str, Any]] = []  # whisper word-level data

#### insight models section #################################################

class ContentInsight(BaseModel):
    """Main model for extracted insights with flexible structure"""
    extraction_goal: str
    source_info: SourceInfo
    
    # core insights with timestamps
    key_insights: List[TimestampedItem] = []
    action_items: List[TimestampedItem] = []
    quotes: List[TimestampedItem] = []
    
    # metadata and processing info
    processing_chunks: int
    total_processing_time: Optional[float] = None
    metadata: Dict[str, Any] = {}
    
    # flexible additional data based on extraction goal
    custom_extractions: Dict[str, List[TimestampedItem]] = {}

#### utility functions section ##############################################

def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format"""
    if seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}:{secs:02d}"
    else:
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}:{mins:02d}:{secs:02d}"

def create_youtube_link(base_url: str, timestamp_seconds: float) -> str:
    """Create clickable YouTube link with timestamp"""
    return f"{base_url}&t={int(timestamp_seconds)}s"