# Fish Speech Fine-tuning Guide

Полное руководство по fine-tuning модели Fish Speech для создания кастомных голосов.

## Содержание

1. [Обзор процесса](#обзор-процесса)
2. [Требования](#требования)
3. [Подготовка данных](#подготовка-данных)
4. [Fine-tuning модели](#fine-tuning-модели)
5. [Использование модели](#использование-модели)
6. [Примеры](#примеры)
7. [Troubleshooting](#troubleshooting)

## Обзор процесса

Fine-tuning Fish Speech состоит из 5 основных этапов:

1. **Подготовка данных** - конвертация аудио в нужный формат
2. **Извлечение семантических токенов** - кодирование аудио в токены
3. **Создание protobuf датасета** - упаковка данных для обучения
4. **Fine-tuning с LoRA** - адаптация модели под ваши данные
5. **Объединение весов** - создание финальной модели

## Требования

### Системные требования

- **Python 3.10+**
- **GPU**: минимум 8GB VRAM (для fine-tuning), 4GB (для inference)
- **RAM**: минимум 16GB
- **Место на диске**: 10-20GB (зависит от размера датасета)
- **Время**: 1-4 часа (зависит от размера датасета и GPU)

### Зависимости

```bash
# Основные зависимости для подготовки данных
poetry run pip install librosa soundfile whisper tqdm yt-dlp

# Для загрузки видео с YouTube (опционально)
pip install yt-dlp
```

### Качество данных

- **Минимум**: 10-30 минут аудио высокого качества
- **Рекомендуется**: 30-60 минут разнообразного контента
- **Формат**: 44.1kHz, mono, WAV/MP3/FLAC
- **Чистота**: минимум фонового шума, четкая речь
- **Текст**: точная транскрипция для каждого аудио файла

## Подготовка данных

### Автоматическая подготовка

```bash
# Обработка локальной папки с аудио
poetry run python prepare_dataset.py \
  --input /path/to/raw/audio \
  --output training_data/my_voice \
  --normalize \
  --split-long \
  --auto-transcribe

# Загрузка и обработка с YouTube (автоматическая сегментация на 10с)
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 10

# Кастомная длительность сегментов (например, 15 секунд)
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 15

# Обработка только первых 20 минут видео
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --max-duration 20

# Комбинация: первые 30 минут + сегменты по 12 секунд
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=VIDEO_ID" \
  --output training_data/youtube_voice \
  --speaker "YouTuber_Name" \
  --youtube \
  --auto-transcribe \
  --whisper-model medium \
  --segment-duration 12 \
  --max-duration 30
```

### Ручная подготовка

Если у вас уже есть подготовленные данные, организуйте их так:

```
training_data/
└── my_voice/
    ├── SPEAKER_NAME/
    │   ├── audio_001.wav
    │   ├── audio_001.lab
    │   ├── audio_002.wav
    │   ├── audio_002.lab
    │   └── ...
    └── dataset_summary.json
```

Где:
- `.wav` файлы содержат аудио сегменты (желательно 10-30 секунд)
- `.lab` файлы содержат точную транскрипцию аудио
- Имена файлов должны совпадать (кроме расширения)

## Fine-tuning модели

### Полный автоматический пайплайн

```bash
# Запуск полного процесса обучения
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_voice \
  --model-version 1.5 \
  --full-pipeline \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 1e-4
```

### Пошаговый процесс

```bash
# Шаг 1: Подготовка данных (если не сделано)
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir raw_audio/ \
  --prepare-data

# Шаг 2: Извлечение семантических токенов
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_custom_voice \
  --extract-tokens \
  --batch-size-extract 16

# Шаг 3: Обучение модели
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --data-dir training_data/my_custom_voice \
  --train \
  --max-steps 200 \
  --batch-size 2 \
  --learning-rate 5e-5

# Шаг 4: Объединение весов
poetry run python finetune_tts.py \
  --project my_custom_voice \
  --merge-weights \
  --data-dir training_data/my_custom_voice
```

### Безопасные параметры обучения

⚠️ **Важно**: Неправильные параметры могут привести к переобучению или нестабильности модели!

| Параметр | "Безопасное" значение | Комментарий |
|----------|----------------------|-------------|
| **VRAM** | ≥ 8 ГБ для fine-tune | Проверено в официальной документации |
| **batch_size** | 2-4 (при 8 ГБ) | С учётом gradient_accumulation_steps легко увеличить «логический» batch |
| **learning_rate** | 1e-5 – 5e-5 | Более высокий LR даёт шум / «провал» после 300 шагов |
| **max_steps** | 100-300 | Хватает, чтобы модель «подхватила» интонацию; дальше — риск переобучения |
| **LoRA rank / α** | r_8_alpha_16 | Предустановленная конфигурация в гайде |

### ⚠️ Важные ограничения Fine-tuning

**По умолчанию fine-tune обучает произношение, но не тембр.**

- **Для тембра нужно**: больше шагов (≈ 500-1000) + разнообразные промпты
- **Риск**: без этого голос «поплывёт» - может стать нестабильным или неестественным
- **Рекомендация**: начинайте с 100-300 шагов для проверки качества, затем увеличивайте при необходимости

**Признаки переобучения:**
- Голос становится роботизированным
- Появляются артефакты и шумы
- Модель теряет естественность речи
- Loss перестаёт снижаться или начинает расти

### Стратегии обучения для разных целей

#### 🎯 Для изучения произношения (быстро и безопасно)
```bash
poetry run python finetune_tts.py \
  --project quick_pronunciation \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 100 \
  --batch-size 2 \
  --learning-rate 2e-5
```
**Результат**: Модель научится базовому произношению за 15-30 минут

#### 🎤 Для захвата тембра голоса (медленно, требует осторожности)
```bash
# Первый этап: базовое обучение
poetry run python finetune_tts.py \
  --project voice_timbre_stage1 \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 300 \
  --batch-size 2 \
  --learning-rate 2e-5

# Второй этап: тонкая настройка тембра
poetry run python finetune_tts.py \
  --project voice_timbre_stage2 \
  --data-dir training_data/my_voice \
  --train \
  --max-steps 500 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --checkpoint-path results/voice_timbre_stage1/checkpoints/
```
**Результат**: Более точное воспроизведение тембра, но риск переобучения

## Использование модели

### После завершения fine-tuning

Обученная модель будет сохранена в:
```
checkpoints/my_custom_voice-merged/
├── model.pth              # Основная модель (~1.2GB)
├── tokenizer.tiktoken     # Токенизатор (~1.6MB)
├── config.json            # Конфигурация модели
└── special_tokens.json    # Специальные токены
```

### Интеграция с CLI TTS

Обновите `cli_tts.py` чтобы использовать вашу модель:

```python
# В функции setup_fish_speech() добавьте путь к вашей модели
custom_model_path = "checkpoints/my_custom_voice-merged"
if Path(custom_model_path).exists():
    print(f"🎤 Используем обученную модель: {custom_model_path}")
    return fish_speech_dir, Path(custom_model_path)
```

**Альтернативно**, укажите путь напрямую при вызове TTS:

```bash
# Используйте полный путь к вашей модели
poetry run python cli_tts.py "Тестируем нашу обученную модель" \
  --model-path checkpoints/test_limited-merged \
  -o output/test_finetuned.wav \
  --play
```

### Тестирование модели

```bash
# Тест с кастомной моделью
poetry run python cli_tts.py "Тестируем нашу обученную модель" \
  --model-path checkpoints/my_custom_voice-merged \
  -o output/test_finetuned.wav \
  --play
```

## Примеры

### Пример 1: Обучение на записях подкаста

```bash
# 1. Загружаем подкаст с YouTube
poetry run python prepare_dataset.py \
  --input "https://youtube.com/watch?v=PODCAST_ID" \
  --output training_data/podcast_host \
  --speaker "PodcastHost" \
  --youtube \
  --auto-transcribe \
  --whisper-model large \
  --split-long

# 2. Запускаем обучение
poetry run python finetune_tts.py \
  --project podcast_voice \
  --data-dir training_data/podcast_host \
  --full-pipeline \
  --max-steps 1500 \
  --batch-size 4
```

### Пример 2: Обучение на аудиокниге

```bash
# 1. Подготавливаем главы аудиокниги
poetry run python prepare_dataset.py \
  --input audiobook_chapters/ \
  --output training_data/narrator \
  --normalize \
  --split-long \
  --auto-transcribe \
  --whisper-model base

# 2. Обучение с меньшим learning rate (для более стабильного голоса)
poetry run python finetune_tts.py \
  --project audiobook_narrator \
  --data-dir training_data/narrator \
  --full-pipeline \
  --max-steps 200 \
  --batch-size 2 \
  --learning-rate 2e-5
```

### Пример 3: Обучение на собственном голосе

```bash
# 1. Записываем свой голос (рекомендуется 30-45 минут разнообразного контента)
# Сохраняем как: my_voice/recordings/session_01.wav, session_02.wav, etc.
# Создаём текстовые файлы: session_01.txt, session_02.txt

# 2. Подготавливаем данные
poetry run python prepare_dataset.py \
  --input my_voice/recordings \
  --output training_data/my_voice \
  --speaker "MyVoice" \
  --normalize \
  --split-long

# 3. Обучение с консервативными настройками
poetry run python finetune_tts.py \
  --project my_personal_voice \
  --data-dir training_data/my_voice \
  --full-pipeline \
  --max-steps 1000 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --device mps  # для Apple Silicon
```

## Мониторинг обучения

### Логи обучения

Логи сохраняются в:
```
fish-speech/results/my_custom_voice/
├── checkpoints/
├── logs/
└── training_configs/
```

### Ключевые метрики

Следите за:
- **Loss**: должен постепенно снижаться
- **Learning Rate**: автоматически адаптируется
- **GPU Memory**: не должна превышать доступную

### Остановка обучения

Обучение можно остановить в любой момент нажатием `Ctrl+C`. Чекпоинты сохраняются каждые 100 шагов.

## Troubleshooting

### Частые проблемы

**1. Out of Memory (OOM)**
```bash
# Уменьшите batch size
--batch-size 1

# Или используйте gradient accumulation
--gradient-accumulation-steps 4
```

**2. Модель не сходится**
```bash
# Уменьшите learning rate
--learning-rate 5e-5

# Увеличьте количество шагов
--max-steps 2000
```

**3. Некачественный синтез**
```bash
# Проверьте качество данных
poetry run python prepare_dataset.py --input training_data/ --output check_data/ --auto-transcribe

# Попробуйте другой checkpoint
--checkpoint-step 500  # вместо последнего
```

**4. Долгое обучение**
```bash
# Используйте меньший датасет для тестирования
--max-steps 500

# Проверьте использование GPU
nvidia-smi  # для NVIDIA
```

### Проверка качества данных

```bash
# Анализ подготовленного датасета
poetry run python -c "
import json
with open('training_data/my_voice/dataset_summary.json') as f:
    summary = json.load(f)
    print(f'Файлов: {summary["total_files"]}')
    print(f'Длительность: {summary["total_duration_minutes"]} мин')
    print(f'Спикеры: {summary["speakers"]}')
"
```

### Оптимизация производительности

**Для Apple Silicon (MPS):**
```bash
--device mps
--batch-size 2  # MPS может быть менее стабильным с большими батчами
```

**Для NVIDIA GPU:**
```bash
--device cuda
--batch-size 4  # или больше если позволяет память
```

**Для CPU (медленно):**
```bash
--device cpu
--batch-size 1
--max-steps 100  # для тестирования
```

## Устранение неполадок

### Проблема: Ошибка типов данных в LoRA слоях

**Симптомы:**
```
RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float
```

**Причина:** 
Предобученная модель Fish Speech загружается с BFloat16 точностью, а LoRA адаптеры инициализируются в Float32. Это приводит к несовместимости типов данных при матричных операциях.

**Решения:**

1. **Автоматическое (рекомендуется):** Скрипт автоматически использует FP32 для всех операций
   ```bash
   # Скрипт автоматически настроит правильную точность
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train
   ```

2. **Если ошибка повторяется:** Принудительно укажите устройство и точность
   ```bash
   # Явная настройка точности
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --device cpu --batch-size 1
   ```

**Технические детали:**
- Базовая модель Fish Speech сохранена с BFloat16 точностью
- LoRA слои по умолчанию создаются в Float32
- Скрипт принудительно использует FP32 для всех компонентов (`trainer.precision=32` + `model.torch_dtype=float32`)
- Это может замедлить обучение, но обеспечивает стабильность

### Проблема: Ошибка тензоров на Apple Silicon (MPS)

**Симптомы:**
```
RuntimeError: Expected scalar_type == ScalarType::Float || inputTensor.scalar_type() == ScalarType::Int || scalar_type == ScalarType::Bool to be true, but got false.
```

**Причина:** 
MPS (Metal Performance Shaders) на Apple Silicon имеет ограничения совместимости с некоторыми операциями PyTorch, особенно с mixed precision training и определёнными типами тензоров.

**Решения:**

1. **Автоматическое (рекомендуется):** Скрипт автоматически переключается на CPU при обнаружении Apple Silicon
   ```bash
   # CPU используется автоматически на Apple Silicon
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train
   ```

2. **Принудительное использование MPS (экспериментально):**
   ```bash
   # Попробовать MPS с полной точностью
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --force-mps
   ```

3. **Явное указание CPU:**
   ```bash
   # Принудительно использовать CPU
   poetry run python finetune_tts.py --project my_voice --data-dir training_data/my_voice --train --device cpu
   ```

**Производительность на Apple Silicon:**
- **CPU:** Стабильно, поддерживает все операции, медленнее GPU
- **MPS:** Быстрее CPU, но может быть нестабильным с некоторыми моделями
- Рекомендуется начать с CPU, а при необходимости попробовать `--force-mps`

### Проблема: Медленное обучение

**Симптомы:** Обучение идёт очень медленно

**Решения:**
1. Уменьшите размер батча: `--batch-size 1`
2. Уменьшите количество шагов: `--max-steps 500`
3. Используйте меньший LoRA rank: `--lora-config r_4_alpha_8`
4. Если есть NVIDIA GPU, используйте `--device cuda`

### Проблема: Нехватка памяти

**Симптомы:** 
```
RuntimeError: [enforce fail at alloc_cpu.cpp] data.
OutOfMemoryError: Unable to allocate array
```

**Решения:**
1. Уменьшите размер батча до 1: `--batch-size 1`
2. Используйте CPU вместо GPU: `--device cpu`
3. Закройте другие приложения
4. Уменьшите количество worker'ов: `--num-workers-extract 1`

## Продвинутые техники

### Смешивание голосов

Можно обучить модель на нескольких голосах:

```bash
# Подготавливаем данные для каждого спикера отдельно
poetry run python prepare_dataset.py --input speaker1_data/ --output training_data/multi_voice --speaker Speaker1
poetry run python prepare_dataset.py --input speaker2_data/ --output training_data/multi_voice --speaker Speaker2

# Обучаем на объединённом датасете
poetry run python finetune_tts.py \
  --project multi_voice_model \
  --data-dir training_data/multi_voice \
  --full-pipeline
```

### Инкрементальное обучение

```bash
# Дообучение существующей модели
poetry run python finetune_tts.py \
  --project my_voice_v2 \
  --data-dir new_training_data/ \
  --train \
  --pretrained-ckpt-path checkpoints/my_custom_voice-merged/
```

### Проверка обученной модели

После завершения обучения проверьте, что все файлы созданы корректно:

```bash
# Проверка размеров файлов модели
ls -lh checkpoints/my_custom_voice-merged/

# Должно показать примерно:
# model.pth              ~1.2GB   # Основная модель
# tokenizer.tiktoken     ~1.6MB   # Токенизатор
# config.json            ~1KB     # Конфигурация
# special_tokens.json    ~30KB    # Специальные токены
```

**Ожидаемые размеры:**
- `model.pth`: 1.0-1.3 GB (зависит от размера базовой модели)
- `tokenizer.tiktoken`: 1-2 MB (словарь токенов)
- `config.json`: менее 1 KB (параметры модели)
- `special_tokens.json`: 20-40 KB (специальные токены)

Если какой-то файл отсутствует или имеет неожиданный размер, проверьте логи обучения.

### Эксперименты с LoRA

```bash
# Более агрессивное обучение
--lora-config r_16_alpha_32

# Более консервативное обучение  
--lora-config r_4_alpha_8
```

## Заключение

Fine-tuning Fish Speech позволяет создавать высококачественные кастомные голоса. Ключевые факторы успеха:

1. **Качественные данные**: чистые записи с точной транскрипцией
2. **Достаточный объём**: минимум 10-30 минут аудио
3. **Правильные параметры**: начинайте с консервативных настроек
4. **Терпение**: хорошие результаты требуют времени

Успешного fine-tuning! 🎤✨