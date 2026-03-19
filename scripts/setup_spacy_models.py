from __future__ import annotations

import subprocess
import sys


REQUIRED_MODELS = [
    "en_core_web_sm",
    "fr_core_news_md",
]


if __name__ == "__main__":
    for model in REQUIRED_MODELS:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
