#!/bin/env python3
import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import srt
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio(input_path: str, output_path: str) -> None:
    """Extract audio from video/audio file and convert to MP3 using FFmpeg."""
    logger.info("Starting audio extraction process")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vn', '-acodec', 'libmp3lame', '-q:a', '2',
            output_path,
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Audio extracted to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio extraction failed: {e}")
        raise

def split_audio(input_path: str, chunk_duration: int, output_dir: str) -> list:
    """Split audio into chunks using FFmpeg."""
    try:
        chunk_pattern = str(Path(output_dir) / "chunk_%03d.mp3")
        cmd = [
            'ffmpeg', '-i', input_path,
            '-f', 'segment', '-segment_time', str(chunk_duration),
            '-c', 'copy', '-y',
            chunk_pattern,
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        chunks = sorted(Path(output_dir).glob("chunk_*.mp3"))
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio splitting failed: {e}")
        raise

def get_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get duration: {e}")
        raise

def transcribe_chunk(client: OpenAI, chunk_path: Path, language: str) -> str:
    """Transcribe a single chunk using OpenAI API."""
    try:
        with open(chunk_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="srt"
            )
        return transcription
    except Exception as e:
        logger.error(f"Transcription failed for {chunk_path}: {e}")
        raise

def process_chunks(chunks: list, api_key: str, language: str) -> list:
    """Process all chunks and return combined subtitles."""
    client = OpenAI(api_key=api_key)
    all_subs = []
    cumulative_offset = 0.0

    for idx, chunk_path in enumerate(chunks):
        logger.info(f"Processing chunk {idx+1}/{len(chunks)}: {chunk_path.name}")
        
        # Get chunk duration
        duration = get_duration(str(chunk_path))
        logger.debug(f"Chunk duration: {duration:.2f}s")

        # Transcribe chunk
        srt_content = transcribe_chunk(client, chunk_path, language)
        
        # Parse and adjust timestamps
        chunk_subs = list(srt.parse(srt_content))
        for sub in chunk_subs:
            sub.start += srt.timedelta(seconds=cumulative_offset)
            sub.end += srt.timedelta(seconds=cumulative_offset)
        
        all_subs.extend(chunk_subs)
        cumulative_offset += duration

    return all_subs

def generate_final_srt(subs: list, output_path: str) -> None:
    """Generate final SRT file with corrected timestamps."""
    subs = srt.sort_and_reindex(subs)
    with open(output_path, 'w') as f:
        f.write(srt.compose(subs))
    logger.info(f"SRT file generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video to SRT with chunking")
    parser.add_argument('input', help="Input audio/video file path")
    parser.add_argument('output', nargs='?', help="Optional output SRT file path (default: same as input with .srt extension)")
    parser.add_argument('--language', required=True, help="Language code (e.g., en, es, fr)")
    parser.add_argument('--chunk-duration', type=int, default=600,
                       help="Chunk duration in seconds (default: 600 = 10 minutes)")
    parser.add_argument('--api-key', help="OpenAI API key (default: uses OPENAI_API_KEY environment variable)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Step 1: Prepare audio
            audio_path = str(Path(tmpdir) / "audio.mp3")
            extract_audio(args.input, audio_path)

            # Step 2: Split audio into chunks
            chunks = split_audio(audio_path, args.chunk_duration, tmpdir)
            if not chunks:
                raise ValueError("No audio chunks created")

            # Step 3: Process all chunks
            api_key = args.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api-key argument")
            subs = process_chunks(chunks, api_key, args.language)

            # Step 4: Generate final SRT
            output_path = args.output or Path(args.input).with_suffix('.srt')
            generate_final_srt(subs, str(output_path))

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {e}")
        exit(1)
