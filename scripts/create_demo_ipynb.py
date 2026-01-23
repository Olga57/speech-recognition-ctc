
from __future__ import annotations
import os
import json

CURRENT_DIR = os.getcwd() 
OUT = os.path.join(CURRENT_DIR, "demo.ipynb")
GITHUB_USER = "Olga57"
REPO_NAME = "speech-recognition-ctc" 

def nb_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def main():
    cells = [
        md_cell(f"# 🎙️ ASR Project Demo (CTC-BiLSTM)\n\nДемонстрация работы системы распознавания речи.\n"),
        
        md_cell("## 1. 🛠️ Установка и Скачивание кода"),
        nb_cell(
            f"import os\nUSER = '{GITHUB_USER}'\nREPO = '{REPO_NAME}'\n\n"
            f"if not os.path.exists(REPO):\n    !git clone https://github.com/{{USER}}/{{REPO}}.git\n\n"
            f"%cd {{REPO}}\n!pip install -q -r requirements.txt\n!pip install -q gdown hydra-core omegaconf\n"
            f"!apt-get install -y libsndfile1 ffmpeg > /dev/null"
        ),
        
        md_cell("## 2. ⚡ Быстрый тест (Librispeech)"),
        nb_cell(
            "import os\n"
            "import sys\n"
            "\n"
            "# 1. Настраиваем пути\n"
            "project_root = os.getcwd()\n"
            "os.environ['PYTHONPATH'] = project_root\n"
            "\n"
            "# 2. ЗАПУСК (С флагами для Hydra)\n"
            "!python inference.py dataset=librispeech \\\n"
            "    +dataset.valid_split=validation.clean \\\n"
            "    +dataset.max_valid_items=5 \\\n"
            "    +dataset.train_source=null \\\n"
            "    +device='cuda' \\\n"
            "    +decode='beam' \\\n"
            "    +out_dir='predictions_libri'\n"
            "\n"
            "!head -n 5 predictions_libri/*.txt"
        ),
        
        md_cell("## 3. 📂 Тест на вашем датасете (Google Drive)"),
        nb_cell(
            "import gdown\nimport os\n"
            "os.environ['PYTHONPATH'] = os.getcwd()\n"
            "url = input('Ссылка на Google Drive (ZIP): ')\n"
            "if 'drive.google.com' in url:\n"
            "    file_id = url.split('/d/')[-1].split('/')[0]\n"
            "    dl_url = f'https://drive.google.com/uc?id={file_id}'\n"
            "    gdown.download(dl_url, 'custom.zip', quiet=False)\n"
            "    !unzip -q -o custom.zip -d custom_data\n"
            "    ds_root = 'custom_data'\n"
            "    for r, d, f in os.walk('custom_data'):\n        if 'audio' in d: ds_root = r\n"
            "    !python inference.py dataset=custom dataset.custom_root=\"{ds_root}\" \\\n"
            "         +device='cuda' +decode='beam' +out_dir='preds_custom' +dataset.train_source=null\n"
            "    !python calc_metrics.py --ref_dir \"{ds_root}/transcriptions\" --hyp_dir \"preds_custom\"\n"
            "else:\n    print('Это не ссылка на Google Drive')"
        )
    ]
    
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("✅ Demo notebook обновлен.")

if __name__ == "__main__":
    main()
