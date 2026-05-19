import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


DEFAULT_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}


def build_palette(max_id: int) -> Dict[int, Tuple[int, int, int]]:
    palette = dict(DEFAULT_PALETTE)
    for idx in range(max_id + 1):
        if idx in palette:
            continue
        # Simple deterministic color for unseen ids
        palette[idx] = ((idx * 37) % 256, (idx * 67) % 256, (idx * 97) % 256)
    return palette


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for value in np.unique(mask):
        color = palette.get(int(value), (255, 255, 255))
        rgb[mask == value] = color
    return rgb


def process_folder(input_dir: Path, output_dir: Path, recursive: bool) -> None:
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.is_file():
            continue
        try:
            mask = np.array(Image.open(path))
        except OSError:
            continue
        if mask.ndim != 2:
            # Skip non-single-channel masks
            continue
        palette = build_palette(int(mask.max()))
        colored = colorize_mask(mask, palette)

        rel_dir = path.parent.relative_to(input_dir)
        out_dir = output_dir / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.with_suffix(".png").name
        Image.fromarray(colored).save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorize mask images for quick review")
    parser.add_argument("--input", "-i", required=True, help="Input masks folder")
    parser.add_argument("--output", "-o", required=True, help="Output folder for colorized masks")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(Path(args.input), Path(args.output), args.recursive)


if __name__ == "__main__":
    main()
