import argparse
import shutil
from pathlib import Path

from PIL import Image


def convert_directory(images_dir: Path, output_dir: Path, recursive: bool) -> None:
    if recursive:
        paths = images_dir.rglob("*")
    else:
        paths = images_dir.glob("*")

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.is_file():
            continue
        rel_path = path.relative_to(images_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as img:
                if img.mode == "L":
                    shutil.copy2(path, out_path)
                    continue
                gray = img.convert("L")
                gray.save(out_path)
        except OSError:
            # Skip files that are not images
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images to grayscale (L)")
    parser.add_argument("--images", "-i", required=True, help="Path to images directory")
    parser.add_argument("--output", "-o", required=True, help="Path to output directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(Path(args.images), Path(args.output), args.recursive)


if __name__ == "__main__":
    main()
