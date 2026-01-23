# ASR Project (CTC-BiLSTM)

Этот репозиторий содержит реализацию системы автоматического распознавания речи (ASR) на базе CTC-модели.

Проект выполнен в рамках учебного курса. Модель обучена на датасете **LibriSpeech**.

---

##  Особенности реализации

- **Архитектура:**
  - Энкодер: 3-слойная **BiLSTM** (Hidden size 256).
  - Препроцессинг: **MelSpectrogram** + **Convolutional Subsampling** для уменьшения временного разрешения.
- **Декодинг:**
  - Реализован **Greedy Search**.
  - Реализован собственный **Beam Search** (на чистом Python) для улучшения качества.
- **Аугментации:**
  - Использован `SpecAugment` (Frequency Masking, Time Masking).
  - `TimeStretch` для изменения скорости речи.
  - Это позволило избежать переобучения на малом датасете.
- **Инструментарий:**
  - Конфигурация через **Hydra**, логирование экспериментов в **WandB**.

---

##  Установка

Для запуска проекта выполните следующие шаги:

1. **Клонирование репозитория:**

```bash
git clone https://github.com/Olga57/speech-recognition-ctc
cd speech-recognition-ctc
```

2. **Установка Python-зависимостей:**

```bash
pip install -r requirements.txt
```

3. **Установка системных библиотек:**
Для работы `torchaudio` и обработки аудиофайлов необходимы `libsndfile` и `ffmpeg`.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
```

---

##  Тестирование и Инференс

В проекте реализован гибкий пайплайн для проверки модели как на стандартных, так и на пользовательских данных.

### 1. Подготовка данных (Custom Dataset)

Для тестирования модели на собственных аудиозаписях (класс `CustomDirDataset`), организуйте папку строго в следующем формате:

```text
MyDataFolder/
├─ audio/
│  ├─ speaker1_001.wav   # Поддерживаются .wav, .flac, .mp3
│  ├─ speaker1_002.wav
│  └─ ...
└─ transcriptions/       # (опционально) Эталонные тексты для подсчета метрик
   ├─ speaker1_001.txt   # Имя файла должно совпадать с аудио
   ├─ speaker1_002.txt
   └─ ...
```

### 2. Запуск Инференса (`inference.py`)

Скрипт загружает веса модели, обрабатывает аудио из указанной папки и сохраняет распознанный текст.

**Пример запуска на кастомном датасете:**

```bash
python inference.py \
  dataset=custom \
  dataset.custom_root="/path/to/MyDataFolder" \
  device="cuda" \
  decode="beam" \
  out_dir="predictions"
```

**Пример запуска на валидации LibriSpeech:**

```bash
python inference.py \
  dataset=librispeech \
  dataset.valid_split="test.clean" \
  device="cuda" \
  decode="beam" \
  out_dir="predictions_libri"
```

**Аргументы:**
- `dataset`: Тип датасета (`custom` или `librispeech`).
- `dataset.custom_root`: Путь к корневой папке кастом датасета.
- `device`: Устройство для вычислений (`cpu` или `cuda`).
- `decode`: Метод декодирования (`greedy` или `beam`).
- `out_dir`: Папка для сохранения результатов (создается автоматически).

### 3. Подсчет метрик (`calc_metrics.py`)

Для оценки качества распознавания (WER/CER) используется отдельный скрипт, который сравнивает папку с предсказаниями и папку с эталонными транскрипциями.

```bash
python calc_metrics.py \
  --ref_dir "/path/to/MyDataFolder/transcriptions" \
  --hyp_dir "predictions"
```

---

##  Демонстрационный ноутбук

Для интерактивного тестирования в репозиторий включён файл `demo.ipynb`. Он адаптирован для запуска в **Google Colab**.

[Открыть Demo Notebook](demo.ipynb)

### Как работает демо

1. **Авто-развертывание:** Скрипт автоматически клонирует этот репозиторий и скачивает веса модели, создавая изолированное окружение.
2. **Интеграция с Google Drive:** Реализована загрузка пользовательских zip-архивов через `gdown`.
3. **Умный поиск:** Скрипт самостоятельно сканирует распакованный архив и находит папку `audio/`, исключая ошибки путей.
4. **Полный цикл:** Автоматически запускается инференс и подсчет метрик одной командой.

---

##  Обучение (Воспроизведение результатов)

Чтобы обучить модель с нуля, используйте скрипт `run_train.py`. Параметры обучения управляются через конфиги Hydra.

```bash
python run_train.py \
  training.device="cuda" \
  training.batch_size=24 \
  training.epochs=50 \
  training.run_name="BiLSTM_Training"
```

### Графики и метрики

В процессе обучения логировались **Loss** и **CER**.

- **Loss:** Стабильное падение с 3.5 до ~0.5.
- **Convergence:** Использование `OneCycleLR` обеспечило быструю сходимость модели.


#### Итоговые результаты на test-clean

| Metric | Greedy Search | Beam Search |
|---|---:|---:|
| CER | ~11.9% | ~11.9% |
| WER | ~38.0% | ~38.0% |

---

##  Структура проекта

```text
.
├── src/
│   ├── asr_datasets/      # Реализация CustomDirDataset и LibrispeechDataset
│   ├── augmentations/     # Аугментации (WaveAugs, SpecAugs)
│   ├── configs/           # Конфигурационные файлы .yaml (Hydra)
│   ├── model/             # Архитектура нейросети (BiLSTM)
│   ├── text/              # Токенизация и Beam Search декодер
│   ├── trainer/           # Логика цикла обучения
│   └── utils/             # Вспомогательные утилиты
├── weights/               # Сохраненные веса модели (best.pt)
├── run_train.py           # Точка входа для обучения
├── inference.py           # Точка входа для инференса
├── calc_metrics.py        # Скрипт подсчета метрик
├── demo.ipynb             # Интерактивное демо
└── requirements.txt       # Зависимости проекта
```
