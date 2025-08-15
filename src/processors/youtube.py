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
            file_size = os.path.getsize(audio_path)
            max_size = 24 * 1024 * 1024  # 24MB limit
            
            if file_size <= max_size:
                # small enough, transcribe directly
                return self._transcribe_single_file(audio_path)
            else:
                # too large, split into chunks and transcribe separately
                return self._transcribe_large_file(audio_path)
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    def _transcribe_single_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe a single audio file"""
        with open(audio_path, 'rb') as audio_file:
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
            
            # convert to dict format for easier manipulation
            segments = []
            if hasattr(response, 'segments'):
                for seg in response.segments:
                    segments.append({
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    })
            
            words = []
            if hasattr(response, 'words'):
                for word in response.words:
                    words.append({
                        'start': word.start,
                        'end': word.end,
                        'word': word.word
                    })
            
            return {
                'text': response.text,
                'segments': segments,
                'words': words,
                'language': getattr(response, 'language', 'unknown'),
                'duration': getattr(response, 'duration', None)
            }
    
    def _transcribe_large_file(self, audio_path: str) -> Dict[str, Any]:
        """Split large audio file and transcribe in chunks"""
        import ffmpeg
        
        # get audio duration
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
        
        # split into 10-minute chunks (should be under 25MB each)
        chunk_duration = 600  # 10 minutes
        chunks = []
        
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        for i, start_time in enumerate(range(0, int(duration), chunk_duration)):
            chunk_path = os.path.join(base_dir, f"{base_name}_chunk_{i:03d}.mp3")
            
            # extract chunk
            (
                ffmpeg
                .input(audio_path, ss=start_time, t=chunk_duration)
                .output(chunk_path, acodec='mp3', audio_bitrate='64k')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # transcribe chunk
            chunk_result = self._transcribe_single_file(chunk_path)
            
            # adjust timestamps to account for chunk offset
            if chunk_result.get('segments'):
                for segment in chunk_result['segments']:
                    segment['start'] += start_time
                    segment['end'] += start_time
            
            if chunk_result.get('words'):
                for word in chunk_result['words']:
                    word['start'] += start_time
                    word['end'] += start_time
            
            chunks.append(chunk_result)
            
            # cleanup chunk file
            os.remove(chunk_path)
        
        # combine all chunks
        combined_text = ' '.join(chunk['text'] for chunk in chunks)
        combined_segments = []
        combined_words = []
        
        for chunk in chunks:
            combined_segments.extend(chunk.get('segments', []))
            combined_words.extend(chunk.get('words', []))
        
        return {
            'text': combined_text,
            'segments': combined_segments,
            'words': combined_words,
            'language': chunks[0].get('language', 'unknown') if chunks else 'unknown',
            'duration': duration
        }
    
    def _compress_audio_if_needed(self, audio_path: str) -> str:
        """Compress audio if it exceeds Whisper's size limit"""
        import ffmpeg
        
        file_size = os.path.getsize(audio_path)
        max_size = 24 * 1024 * 1024  # 24MB to stay under 25MB limit
        
        if file_size <= max_size:
            return audio_path
        
        # create compressed version
        compressed_path = audio_path.rsplit('.', 1)[0] + '_compressed.mp3'
        
        try:
            # compress audio: reduce bitrate and convert to mp3
            (
                ffmpeg
                .input(audio_path)
                .output(compressed_path, acodec='mp3', audio_bitrate='64k')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # verify the compressed file is smaller
            compressed_size = os.path.getsize(compressed_path)
            if compressed_size < max_size:
                return compressed_path
            else:
                # if still too large, use original and let it fail with better error
                return audio_path
                
        except Exception as e:
            print(f"Warning: Audio compression failed: {e}")
            return audio_path

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