# Озвучка текста нейросетями

Python-проект для озвучки длинных русскоязычных текстов. Он разбивает текст на фрагменты, преобразует числа в слова, синтезирует речь и собирает итоговый WAV с паузами. По длительности каждого фрагмента строятся субтитры SRT. Из аудио и субтитров можно собрать вертикальное видео.

- `main.py` использует Silero v5 с русским голосом `eugene`. PyTorch скачивает модель при первом запуске.
- `main_qwen.py` использует Qwen3-TTS и принимает аудиообразец голоса вместе с его расшифровкой.
- [Ноутбук Coqui XTTS](notebooks/coqui_multisampling.ipynb) показывает эксперименты с озвучкой перекрывающихся отрывков и выравниванием звуковых волн.

## Запуск

Понадобится Python 3.11. Для видео установите FFmpeg с фильтром `subtitles` (libass).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py text.txt --output out
```

`text.txt` — ваш текстовый файл в UTF-8. В `out` появятся `audiobook.wav`, `audiobook.srt` с субтитрами по фрагментам и `book_pages.srt` со сгруппированными субтитрами. Добавьте `--video`, чтобы получить `book.mp4`.

Для варианта с Qwen установите `qwen-tts` и передайте свой аудиообразец и расшифровку:

```bash
python -m pip install qwen-tts
python main_qwen.py text.txt --output out/qwen \
  --reference-audio reference.wav --reference-text "Расшифровка вашего аудиообразца"
```

Для экспериментов Coqui установите `coqui-tts`, `jupyterlab`, `numpy`, `matplotlib`, `librosa` и `scipy` в отдельном окружении.
