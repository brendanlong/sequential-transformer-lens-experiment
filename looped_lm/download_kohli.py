"""Fetch Kohli et al.'s two-hop data and systematicity checkpoints from Google Drive.

uv run --group looped python -m looped_lm.download_kohli ext/kohli
"""

import argparse
from pathlib import Path
from typing import Any

import gdown

FOLDER = "https://drive.google.com/drive/folders/1AQvp8erdtRRn9_ix7E1ssmyWarwFNLVU"
WANTED = ("data/composition.2000.200.7.2/", "checkpoints/systematicity/")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    files: list[Any] = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
    for f in files:
        dst = root / f.path
        if not f.path.startswith(WANTED) or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(dst.suffix + ".part")
        gdown.download(id=f.id, output=str(part), quiet=True)
        part.rename(dst)
        print(dst, flush=True)


if __name__ == "__main__":
    main()
