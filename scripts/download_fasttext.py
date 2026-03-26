from pathlib import Path
import shutil
import gzip
import requests
from tqdm import tqdm

OUTPUT_DIR = Path("models")

URLS = [
    #"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz",
]


def download_model(url, out_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("content-length", 0))

        with gzip.GzipFile(fileobj=r.raw) as gz, \
                open(out_path, "wb") as f_out, \
                tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading+Extracting") as pbar:

            while True:
                chunk = gz.read(8192)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        filename = url.split("/")[-1]
        gz_path = OUTPUT_DIR / filename
        bin_path = OUTPUT_DIR / filename.replace(".gz", "")

        # Download if not already present
        if not gz_path.exists():
            download_model(url, bin_path)
        else:
            print(f"{gz_path} already exists, skipping download.")
