# Fish Speech E2E Tests

Автоматизированные тесты для проверки полного цикла обучения Fish Speech с эмоциональными токенами.

## Что тестируется

- ✅ **Обучение с нуля** - создание новой модели с эмоциональными токенами
- ✅ **Resume обучения** - продолжение обучения с чекпойнта
- ✅ **Реальные русскоязычные данные** - используются настоящие голоса из проекта
- ✅ **Эмоциональные токены** - автоматическое добавление к русскому тексту
- ✅ **Создание чекпойнтов** - сохранение и загрузка состояния модели
- ✅ **Инференс с обученными моделями** - генерация речи с использованием `flash_optimized_cli.py`
- ✅ **Автопроигрывание аудио** - проверка качества генерации с флагом `--play`
- ✅ **HTML отчеты** - детальные pytest отчеты с автооткрытием в браузере
- ✅ **Системные требования** - проверка MPS/CUDA/CPU

## Структура тестов

```
fs-python/
├── Makefile                # Команды для управления тестами
├── pyproject.toml          # Poetry конфигурация с тестовыми зависимостями
└── tests/
├── test_e2e_training.py    # Основной E2E тест
├── conftest.py             # Конфигурация pytest
├── pytest.ini             # Настройки pytest
├── requirements.txt        # Зависимости для тестов (pip fallback)
├── cleanup.py              # Утилита очистки тестовых данных
├── data/                   # Изолированные тестовые данные
│   ├── real_russian/       # Копии реальных русскоязычных голосов
│   │   ├── RU_Google_Female_Zephyr/
│   │   ├── RU_Male_Goblin_Puchkov/
│   │   └── RU_Google_Male_Achird/
│   └── prepared/           # Подготовленные данные для обучения
└── README.md              # Эта документация
```

## Установка зависимостей

### Установка с Poetry (рекомендуется)
```bash
cd fs-python
poetry install --with test
```

### Установка с pip (альтернатива)
```bash
cd fs-python
pip install -r tests/requirements.txt
```

## Запуск тестов

### Полный E2E тест
```bash
cd fs-python
poetry run python tests/test_e2e_training.py
```

### Через pytest с Poetry
```bash
cd fs-python
poetry run pytest tests/test_e2e_training.py -v
```

### Только быстрые тесты (исключить E2E)
```bash
poetry run pytest -m "not slow" -v
```

### С дополнительным логированием
```bash
poetry run pytest tests/test_e2e_training.py -v -s
```

### Через Makefile (рекомендуется)
```bash
cd fs-python

# Основные команды
make help              # Показать все доступные команды
make install           # Установить зависимости с тестами
make test              # Запустить полный E2E тест
make test-pytest       # Запустить через pytest
make test-html         # Запустить тест с HTML отчетом
make test-with-report  # Полный тест + HTML отчет + автооткрытие
make cleanup           # Очистить тестовые данные

# Дополнительные команды
make test-verbose      # Тест с подробным выводом
make check-system      # Проверить системные требования
make check-gpu         # Проверить GPU/MPS доступность
make show-report       # Показать последний JSON отчет
make open-report       # Открыть HTML отчет в браузере
make quick-test        # Установить + запустить тест
```

### Если Poetry не используется
```bash
cd fs-python
python tests/test_e2e_training.py
pytest tests/test_e2e_training.py -v
```

## Параметры теста

Тест настроен на минимальные ресурсы:
- **Шаги обучения**: 5 (начальное) + 3 (resume) = 8 шагов всего
- **Batch size**: 1 
- **Learning rate**: 1e-4
- **Сохранение**: каждые 3 шага
- **Данные**: 3 реальных русскоязычных голоса с эмоциональными токенами
  - `RU_Google_Female_Zephyr` - женский голос Google (радость, удивление)
  - `RU_Male_Goblin_Puchkov` - мужской голос Гоблина (гнев, грусть)
  - `RU_Google_Male_Achird` - мужской голос Google (радость, страх)

## Ожидаемые результаты

### Успешный тест создаст:
1. **Чекпойнты** (~71MB каждый):
   - `/checkpoints/e2e_test_initial/checkpoints/step_*.ckpt`
   - `/checkpoints/e2e_test_resume/checkpoints/step_*.ckpt`

2. **Аудио файлы** (инференс):
   - `tests/data/inference_outputs/initial_test_*_*.wav`
   - `tests/data/inference_outputs/resume_test_*_*.wav`
   - Размер: 250-360KB каждый файл

3. **Отчеты**:
   - **JSON**: `tests/test_report.json` (детальные метрики)
   - **HTML**: `tests/report.html` (интерактивный отчет pytest)
   - **pytest JSON**: `tests/report.json` (стандартный pytest отчет)

4. **Подготовленные данные**:
   - `tests/data/prepared/` - данные для обучения
   - `tests/data/real_russian/` - реальные русские голоса

### Время выполнения
- **Полный E2E тест с инференсом**: ~6-8 минут на Apple Silicon
- **Только обучение**: 2-3 минуты
- **Инференс**: ~45 секунд на тест (6 тестов = ~5 минут)
- **Максимум**: 30 минут (таймаут)

## Эмоциональные токены

Тест проверяет работу с токенами:
- `(joyful)` - радость
- `(sad)` - грусть  
- `(angry)` - гнев
- `(scared)` - страх
- `(surprised)` - удивление

## Инференс тестирование

### Что тестируется
- **2 чекпойнта**: initial (5 шагов) и resume (8 шагов)
- **3 тестовых фразы** с разными эмоциями:
  - `(joyful) Привет! Это тест обученной модели.`
  - `(angry) Какая же бесполезная затея!`
  - `(surprised) Неожиданно, но это работает!`
- **3 русских голоса**: Female Zephyr, Male Goblin, Male Achird

### Результаты последнего теста
- ✅ **Всего тестов**: 6 (2 чекпойнта × 3 фразы)
- ✅ **Успешных**: 6/6 (100% success rate)
- ✅ **Среднее время**: 45.1 секунды на тест
- ✅ **Аудио файлы**: 6 WAV файлов (250-360KB каждый)
- ✅ **Автопроигрывание**: включено с флагом `--play`

## Устранение неполадок

### Тест падает на обучении
```bash
# Быстрая проверка системы через Makefile
make check-system

# Проверка GPU/MPS
make check-gpu

# Или ручная проверка
poetry run python -c "from finetune_tts import check_requirements; print(check_requirements())"
poetry run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Нехватка памяти
- Используется batch_size=1 и minimal settings
- На Apple Silicon нужно ~8GB свободной памяти
- Закройте другие приложения

### Чекпойнты не создаются
- Проверьте права доступа к `/checkpoints/`
- Убедитесь что `save_every_n_steps` < `max_steps`

## Очистка после тестов

Тест **НЕ удаляет** созданные данные автоматически.

### Автоматическая очистка
```bash
cd fs-python

# Через Makefile (рекомендуется)
make cleanup

# Или напрямую
poetry run python tests/cleanup.py
```

### Ручная очистка
```bash
# Удалить тестовые чекпойнты
rm -rf /Users/a1/Project/OPG/checkpoints/e2e_test_*

# Удалить подготовленные данные  
rm -rf tests/data/prepared/

# Удалить отчеты
rm -f tests/test_report.json
```

## Интеграция в CI/CD

### GitHub Actions (базовый)
```yaml
- name: Install dependencies
  run: |
    cd fs-python
    poetry install --with test

- name: Run E2E Tests with HTML report
  run: |
    cd fs-python
    make test-html

- name: Upload test artifacts
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-results
    path: |
      fs-python/tests/report.html
      fs-python/tests/report.json
      fs-python/tests/data/inference_outputs/*.wav
```

### GitHub Actions (расширенный с артефактами)
```yaml
- name: Run full E2E workflow
  run: |
    cd fs-python
    poetry run pytest tests/test_e2e_training.py -v \
      --html=tests/report.html --self-contained-html \
      --json-report --json-report-file=tests/report.json \
      --timeout=2400

- name: Publish test report
  uses: peaceiris/actions-gh-pages@v3
  if: always()
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: fs-python/tests
    destination_dir: test-reports
``` 