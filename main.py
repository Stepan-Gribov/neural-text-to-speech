import argparse
from pathlib import Path

import soundfile as sf
import torch

from audio_pipeline import render_book


def main():
    parser = argparse.ArgumentParser(description="Озвучить русскоязычный текст")
    parser.add_argument("input", type=Path, help="Текстовый файл UTF-8")
    parser.add_argument("--output", type=Path, default=Path("out"), help="Папка результатов")
    parser.add_argument("--video", action="store_true", help="Дополнительно собрать MP4 с субтитрами")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Укажите существующий текстовый файл")
    parts_dir = args.output / "wav_segments"
    parts_dir.mkdir(parents=True, exist_ok=True)
    SPEAKER = "eugene"
    SAMPLE_RATE = 48000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_ru",
        trust_repo=True,
    )
    model.to(device)

    def generate_tts(phrase: str, index: int) -> str:
        try:
            print(f"Синтез фрагмента {index + 1}")
            audio = model.apply_tts(
                text=phrase,
                speaker=SPEAKER,
                sample_rate=SAMPLE_RATE,
                put_accent=True,
            )
            if hasattr(audio, "is_cuda") and audio.is_cuda:
                audio = audio.cpu()
            path = str(parts_dir / f"part_{index:04d}.wav")
            sf.write(path, audio.numpy(), SAMPLE_RATE)
            return path
        except Exception as e:
            print(f"[{index:04d}] skipped: {e}")
            raise RuntimeError(f"Не удалось озвучить фрагмент {index + 1}") from e

    render_book(args.input, args.output, generate_tts, video=args.video)


if __name__ == "__main__":
    main()
