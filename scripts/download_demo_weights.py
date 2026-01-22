from __future__ import annotations

import os
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin"
FILENAME = "demo_wav2vec2.bin"

def resolve_project_root() -> Path:
    # В Colab это всегда /content/asr_project, но оставляем fallback
    colab_root = Path("/content/asr_project")
    if colab_root.exists():
        return colab_root.resolve()
    env_root = os.environ.get("ASR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()

def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[OK] Demo weights already exist: {out_path}")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    print("[DL] Downloading demo ASR weights...")
    print(f"     URL : {url}")
    print(f"     OUT : {out_path}")

    try:
        urllib.request.urlretrieve(url, tmp_path)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists() and (not out_path.exists()):
            try:
                tmp_path.unlink()
            except Exception:
                pass

    print(f"[OK] Saved to: {out_path}")

def main() -> None:
    root = resolve_project_root()
    out_path = root / "weights" / FILENAME
    print(f"[INFO] Project root: {root}")
    download(URL, out_path)

if __name__ == "__main__":
    main()
