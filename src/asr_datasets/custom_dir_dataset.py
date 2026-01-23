
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class CustomDirDataset(Dataset):
    def __init__(self, custom_root, **kwargs):
        self.root = Path(custom_root)
        self.files = []
        # Ищем аудиофайлы всех форматов
        extensions = ['*.wav', '*.flac', '*.mp3']
        if self.root.exists():
            for ext in extensions:
                self.files.extend(list(self.root.rglob(ext)))
        else:
            print(f"Warning: Directory {custom_root} not found.")

    def __getitem__(self, index):
        path = self.files[index]
        audio, sr = torchaudio.load(path)
        # Возвращаем словарь, который ожидает inference.py
        return {
            "audio": audio,
            "text": "",  # Пустой текст, так как мы только предсказываем
            "sample_rate": sr,
            "audio_path": str(path)
        }

    def __len__(self):
        return len(self.files)
