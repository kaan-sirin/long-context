import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv

from ..models.content_insight import SourceInfo
from datetime import datetime

#### youtube processor section ##############################################

class YouTubeProcessor:
    """Handles YouTube video processing with Whisper transcription"""
    
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # yt-dlp configuration for audio extraction
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': '%(title)s.%(ext)s',
            'noplaylist': True,
        }
    
    def process_video(self, video_url: str) -> Dict[str, Any]:
        """
        Process YouTube video: extract audio and transcribe with timestamps
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict containing transcript data and source info
        """
        # extract video metadata first
        video_info = self._get_video_info(video_url)
        
        # download audio to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._download_audio(video_url, temp_dir)
            
            # transcribe with word-level timestamps
            transcript_data = self._transcribe_audio(audio_path)
            
        # create source info
        source_info = SourceInfo(
            source_type="youtube",
            title=video_info.get('title', 'Unknown'),
            url=video_url,
            duration_seconds=video_info.get('duration'),
            processed_at=datetime.now()
        )
        
        return {
            'transcript': transcript_data,
            'source_info': source_info.dict(),
            'video_metadata': video_info
        }
    
    def _get_video_info(self, video_url: str) -> Dict[str, Any]:
        """Extract video metadata without downloading"""
        ydl_opts_info = {**self.ydl_opts, 'skip_download': True}
        
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            try:
                info = ydl.extract_info(video_url, download=False)
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'description': info.get('description', '')[:500] + '...'  # truncated
                }
            except Exception as e:
                raise ValueError(f"Failed to extract video info: {str(e)}")
    
    def _download_audio(self, video_url: str, output_dir: str) -> str:
        """Download audio from YouTube video"""
        ydl_opts = {
            **self.ydl_opts,
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # extract info to get the actual filename
                info = ydl.extract_info(video_url, download=False)
                filename = ydl.prepare_filename(info)
                
                # download the audio
                ydl.download([video_url])
                
                # find the downloaded file (yt-dlp might change extension)
                base_path = Path(filename).with_suffix('')
                for ext in ['.mp3', '.m4a', '.webm', '.opus']:
                    potential_path = str(base_path) + ext
                    if os.path.exists(potential_path):
                        return potential_path
                
                # fallback: find any audio file in the directory
                audio_files = [f for f in os.listdir(output_dir) 
                              if f.lower().endswith(('.mp3', '.m4a', '.webm', '.opus'))]
                if audio_files:
                    return os.path.join(output_dir, audio_files[0])
                
                raise FileNotFoundError("Downloaded audio file not found")
                
            except Exception as e:
                raise ValueError(f"Failed to download audio: {str(e)}")
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper with word-level timestamps"""
        try:
            with open(audio_path, 'rb') as audio_file:
                # use whisper with word-level timestamps
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
                
                return {
                    'text': response.text,
                    'segments': response.segments,
                    'words': getattr(response, 'words', []),  # word-level timestamps
                    'language': getattr(response, 'language', 'unknown'),
                    'duration': getattr(response, 'duration', None)
                }
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")

#### utility functions section ##############################################

def extract_video_id(video_url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats"""
    import re
    
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    
    return None

def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL"""
    return any(domain in url.lower() for domain in ['youtube.com', 'youtu.be'])