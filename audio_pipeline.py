"""Shared text preparation, audio assembly, and subtitle generation."""

import re
import subprocess
import textwrap
from datetime import timedelta

import srt
from num2words import num2words
from pydub import AudioSegment

CHUNK_LIMIT = 1000
PAUSE_MS = 400
LOW_PASS_HZ = 4000
GAIN_DB = -1
PAGE_LIMIT = 1000


def split_text(text, limit=CHUNK_LIMIT):
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) <= limit:
            yield sentence
            continue

        buffer = ""
        for clause in re.split(r"(?<=,)\s+", sentence):
            candidate = f"{buffer} {clause}".strip()
            if len(candidate) <= limit:
                buffer = candidate
                continue
            if buffer:
                yield buffer
            pieces = textwrap.wrap(clause, width=limit)
            yield from pieces[:-1]
            buffer = pieces[-1] if pieces else ""
        if buffer:
            yield buffer


def normalize_russian(text):
    text = re.sub(r"\b\d+\b", lambda match: num2words(int(match[0]), lang="ru"), text)
    return re.sub(r"\bСтатья\b", "СТАТЬЯ", text)


def subtitle_pages(subtitles):
    start = subtitles[0].start
    paragraphs = []
    length = 0
    pages = []
    for subtitle in subtitles:
        if paragraphs and length + 1 + len(subtitle.content) > PAGE_LIMIT:
            pages.append(srt.Subtitle(len(pages) + 1, start, subtitle.start, " ".join(paragraphs)))
            start = subtitle.start
            paragraphs = []
            length = 0
        length += len(subtitle.content) + bool(paragraphs)
        paragraphs.append(subtitle.content)
    pages.append(srt.Subtitle(len(pages) + 1, start, subtitles[-1].end, " ".join(paragraphs)))
    return pages


def render_book(input_file, output, generate_audio, *, video):
    text = input_file.read_text(encoding="utf-8")
    chunks = list(split_text(text))
    if not chunks:
        raise ValueError("Входной файл пуст")

    combined = AudioSegment.empty()
    subtitles = []
    start_ms = 0
    for index, chunk in enumerate(chunks):
        pause = AudioSegment.silent(duration=PAUSE_MS)
        part = AudioSegment.from_wav(generate_audio(normalize_russian(chunk), index))
        end_ms = start_ms + len(part)
        subtitles.append(
            srt.Subtitle(index + 1, timedelta(milliseconds=start_ms), timedelta(milliseconds=end_ms), chunk)
        )
        combined += part + pause
        start_ms = end_ms + len(pause)

    (output / "audiobook.srt").write_text(srt.compose(subtitles), encoding="utf-8")
    (output / "book_pages.srt").write_text(srt.compose(subtitle_pages(subtitles)), encoding="utf-8")
    combined.low_pass_filter(LOW_PASS_HZ).apply_gain(GAIN_DB).export(output / "audiobook.wav", format="wav")

    if video:
        style = r"FontName=Arial\,FontSize=8\,BorderStyle=3\,Outline=2\,Shadow=0\,Alignment=8"
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", "color=c=black:s=720x1280",
                "-i", "audiobook.wav",
                "-vf", f"subtitles=book_pages.srt:force_style='{style}'",
                "-shortest",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y", "book.mp4",
            ],
            cwd=output,
            check=True,
        )
