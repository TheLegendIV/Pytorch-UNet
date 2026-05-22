#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def clear_dir(target: Path) -> None:
    if not str(target):
        raise ValueError("Refusing to clear empty path")
    target.mkdir(parents=True, exist_ok=True)
    for entry in target.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def clear_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        clear_dir(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear contents of directories")
    parser.add_argument("paths", nargs="+", help="Directories to clear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_dirs([Path(p) for p in args.paths])


if __name__ == "__main__":
    main()
