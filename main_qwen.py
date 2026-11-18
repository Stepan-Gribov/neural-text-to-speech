import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from audio_pipeline import render_book


def main():
    parser = argparse.ArgumentParser(description="Озвучить русскоязычный текст")
    parser.add_argument("input", type=Path, help="Текстовый файл UTF-8")
    parser.add_argument("--output", type=Path, default=Path("out"), help="Папка результатов")
    parser.add_argument("--video", action="store_true", help="Дополнительно собрать MP4 с субтитрами")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Укажите существующий текстовый файл")
    if not args.reference_audio.is_file():
        parser.error("Укажите существующий аудиофайл для образца голоса")
    parts_dir = args.output / "wav_segments"
    parts_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(217950)

    if torch.cuda.is_available():
        DEVICE = "cuda"
        DTYPE = torch.float16
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        DTYPE = torch.bfloat16
    else:
        DEVICE = "cpu"
        DTYPE = torch.float32

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="sdpa",
    )


    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(args.reference_audio),
        ref_text=args.reference_text,
        x_vector_only_mode=False,
    )

    def generate_tts(phrase: str, index: int) -> str:
        try:
            print(f"Синтез фрагмента {index + 1}")
            audio, SAMPLE_RATE = model.generate_voice_clone(
                text=phrase,
                language="russian",
                voice_clone_prompt=prompt_items,
                top_k=20,
                top_p=0.93,
                temperature=0.32,
                repetition_penalty=1.2
            )
            if hasattr(audio, "is_cuda") and audio.is_cuda:
                audio = audio.cpu()
            path = str(parts_dir / f"part_{index:04d}.wav")
            sf.write(path, audio[0], SAMPLE_RATE)
            return path
        except Exception as e:
            print(f"[{index:04d}] skipped: {e}")
            raise RuntimeError(f"Не удалось озвучить фрагмент {index + 1}") from e

    render_book(args.input, args.output, generate_tts, video=args.video)


if __name__ == "__main__":
    main()
