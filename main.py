#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from src.processors.youtube import YouTubeProcessor, is_youtube_url
from src.core.chunker import SemanticChunker
from src.core.extractor import InsightExtractor

#### cli interface section ##################################################

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="Extract insights from long-form content (YouTube, podcasts, articles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --url "https://youtube.com/watch?v=xyz" --goal "extract book recommendations"
  python main.py --file podcast.mp3 --goal "find startup ideas"
  python main.py --text article.txt --goal "summarize key points"
        """
    )
    
    # input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--url', help='YouTube video URL')
    input_group.add_argument('--file', help='Audio file path (MP3, WAV, M4A)')
    input_group.add_argument('--text', help='Text file path')
    
    # extraction configuration
    parser.add_argument('--goal', required=True, 
                       help='Extraction goal (e.g., "extract book recommendations")')
    parser.add_argument('--output', help='Output directory (default: ./outputs)')
    parser.add_argument('--markdown', action='store_true', 
                       help='Generate markdown report')
    
    args = parser.parse_args()
    
    # setup output directory
    output_dir = Path(args.output) if args.output else Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    
    try:
        if args.url:
            if not is_youtube_url(args.url):
                console.print("[red]Error: Only YouTube URLs are currently supported[/red]")
                sys.exit(1)
            result = process_youtube_video(args.url, args.goal)
        elif args.file:
            console.print("[yellow]Audio file processing not yet implemented[/yellow]")
            sys.exit(1)
        elif args.text:
            console.print("[yellow]Text file processing not yet implemented[/yellow]")
            sys.exit(1)
        
        # save results
        output_file = save_insights(result, output_dir, args.goal)
        console.print(f"[green]✓ Insights saved to: {output_file}[/green]")
        
        # generate markdown if requested
        if args.markdown:
            markdown_file = generate_markdown_report(result, output_dir)
            console.print(f"[green]✓ Markdown report: {markdown_file}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

#### processing functions section ###########################################

def process_youtube_video(video_url: str, extraction_goal: str):
    """Process YouTube video and extract insights"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        # step 1: download and transcribe
        task1 = progress.add_task("Downloading and transcribing video...", total=None)
        processor = YouTubeProcessor()
        video_data = processor.process_video(video_url)
        progress.remove_task(task1)
        
        # step 2: chunk content
        task2 = progress.add_task("Chunking content for processing...", total=None)
        chunker = SemanticChunker()
        chunks = chunker.chunk_transcript(video_data['transcript'])
        progress.remove_task(task2)
        
        console.print(f"[blue]ℹ Created {len(chunks)} chunks for processing[/blue]")
        
        # step 3: extract insights
        task3 = progress.add_task("Extracting insights with AI...", total=None)
        extractor = InsightExtractor()
        insights = extractor.extract_insights(
            chunks, extraction_goal, video_data['source_info']
        )
        progress.remove_task(task3)
    
    console.print(Panel.fit(
        f"[green]Processing Complete![/green]\n\n"
        f"📹 Video: {insights.source_info.title}\n"
        f"🎯 Goal: {extraction_goal}\n"
        f"📊 Found: {len(insights.key_insights)} insights, "
        f"{len(insights.action_items)} actions, {len(insights.quotes)} quotes\n"
        f"⏱️  Chunks processed: {insights.processing_chunks}",
        title="Results"
    ))
    
    return insights

#### output functions section ###############################################

def save_insights(insights, output_dir: Path, goal: str):
    """Save insights as YAML file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    goal_slug = goal.replace(' ', '_').lower()[:30]
    filename = f"{timestamp}_{goal_slug}.yaml"
    
    output_file = output_dir / filename
    
    with open(output_file, 'w') as f:
        yaml.dump(insights.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    return output_file

def generate_markdown_report(insights, output_dir: Path):
    """Generate markdown report with clickable timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_report.md"
    output_file = output_dir / filename
    
    with open(output_file, 'w') as f:
        f.write(f"# Content Analysis Report\n\n")
        f.write(f"**Source:** {insights.source_info.title}\n")
        f.write(f"**URL:** {insights.source_info.url}\n")
        f.write(f"**Extraction Goal:** {insights.extraction_goal}\n")
        f.write(f"**Processed:** {insights.source_info.processed_at}\n\n")
        
        if insights.key_insights:
            f.write("## Key Insights\n\n")
            for insight in insights.key_insights:
                if insight.source_url and insight.timestamp_display:
                    f.write(f"- [{insight.timestamp_display}]({insight.source_url}) - {insight.content}\n")
                else:
                    f.write(f"- {insight.content}\n")
            f.write("\n")
        
        if insights.action_items:
            f.write("## Action Items\n\n")
            for action in insights.action_items:
                if action.source_url and action.timestamp_display:
                    f.write(f"- [{action.timestamp_display}]({action.source_url}) - {action.content}\n")
                else:
                    f.write(f"- {action.content}\n")
            f.write("\n")
        
        if insights.quotes:
            f.write("## Notable Quotes\n\n")
            for quote in insights.quotes:
                if quote.source_url and quote.timestamp_display:
                    f.write(f"> [{quote.timestamp_display}]({quote.source_url}) \"{quote.content}\"\n\n")
                else:
                    f.write(f"> \"{quote.content}\"\n\n")
    
    return output_file

if __name__ == '__main__':
    main()
