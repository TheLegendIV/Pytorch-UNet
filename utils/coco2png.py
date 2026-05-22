import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(images: List[Dict]) -> Dict[int, Dict]:
    return {img["id"]: img for img in images}


def build_annotations_index(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    by_image: Dict[int, List[Dict]] = {}
    for ann in annotations:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return by_image


def polygon_to_mask(polygons: List[List[float]], width: int, height: int, fill_value: int) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        if len(poly) < 6:
            continue
        points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(points, fill=fill_value)
    return np.array(mask_img, dtype=np.uint8)


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
        # Deterministic color for unseen ids
        palette[idx] = ((idx * 37) % 256, (idx * 67) % 256, (idx * 97) % 256)
    return palette


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for value in np.unique(mask):
        color = palette.get(int(value), (255, 255, 255))
        rgb[mask == value] = color
    return rgb


def parse_groups(groups_text: Optional[str]) -> Dict[int, int]:
    if not groups_text:
        return {}
    mapping: Dict[int, int] = {}
    groups = [g.strip() for g in groups_text.split(";") if g.strip()]
    for group_index, group in enumerate(groups, start=1):
        for item in group.split(","):
            item = item.strip()
            if not item:
                continue
            mapping[int(item)] = group_index
    return mapping


def write_masks(
    coco_json: Path,
    images_dir: Path,
    masks_dir: Path,
    binary: bool,
    category_map: Dict[int, int],
) -> None:
    coco = load_coco(coco_json)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_index = build_image_index(images)
    ann_index = build_annotations_index(annotations)

    masks_dir.mkdir(parents=True, exist_ok=True)

    for image_id, img_info in image_index.items():
        file_name = img_info["file_name"]
        img_path = images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        width = int(img_info.get("width", 0))
        height = int(img_info.get("height", 0))
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in ann_index.get(image_id, []):
            segmentation = ann.get("segmentation", [])
            category_id = int(ann.get("category_id", 0))
            if binary:
                fill_value = 255
            elif category_map:
                fill_value = category_map.get(category_id, 0)
            else:
                fill_value = category_id
            if isinstance(segmentation, list):
                mask_part = polygon_to_mask(segmentation, width, height, fill_value)
                mask = np.maximum(mask, mask_part)

        out_path = masks_dir / Path(file_name).with_suffix(".png").name
        if binary:
            Image.fromarray(mask).save(out_path)
        else:
            palette = build_palette(int(mask.max()))
            colored = colorize_mask(mask, palette)
            Image.fromarray(colored).save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ARCADE COCO polygon annotations to mask PNGs")
    parser.add_argument("--annotations", "-a", required=True, help="Path to COCO JSON (e.g., train.json)")
    parser.add_argument("--images", "-i", required=True, help="Path to images directory")
    parser.add_argument("--masks", "-m", required=True, help="Path to output masks directory")
    parser.add_argument(
        "--groups",
        help="Semicolon-separated groups of category_ids, e.g. '0,1,2;3,4;5,6'",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary masks (foreground=255) instead of class IDs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_map = parse_groups(args.groups)
    write_masks(Path(args.annotations), Path(args.images), Path(args.masks), args.binary, category_map)


if __name__ == "__main__":
    main()
