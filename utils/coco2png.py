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


def polygon_to_mask(
    polygons: List[List[float]],
    width: int,
    height: int,
    fill_value: int,
) -> np.ndarray:
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
        palette[idx] = (
            (idx * 37) % 256,
            (idx * 67) % 256,
            (idx * 97) % 256,
        )

    return palette


def colorize_mask(
    mask: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for value in np.unique(mask):
        color = palette.get(int(value), (255, 255, 255))
        rgb[mask == value] = color

    return rgb


def parse_groups(groups_text: Optional[str]) -> Dict[int, int]:
    """
    Converts a group string like:

        "0,1,2;3,4;5,6"

    into:

        {
            0: 1,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
        }

    This maps original category IDs to new grouped class IDs.
    """
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


def get_fill_value(
    category_id: int,
    binary: bool,
    category_map: Dict[int, int],
) -> int:
    if binary:
        return 255

    if category_map:
        return category_map.get(category_id, 0)

    return category_id


def write_masks(
    coco_json: Path,
    images_dir: Path,
    masks_dir: Path,
    binary: bool,
    single_channel: bool,
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

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions for {file_name}: "
                f"width={width}, height={height}"
            )

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in ann_index.get(image_id, []):
            segmentation = ann.get("segmentation", [])
            category_id = int(ann.get("category_id", 0))

            fill_value = get_fill_value(
                category_id=category_id,
                binary=binary,
                category_map=category_map,
            )

            if isinstance(segmentation, list):
                mask_part = polygon_to_mask(
                    polygons=segmentation,
                    width=width,
                    height=height,
                    fill_value=fill_value,
                )

                # If polygons overlap, the larger class ID/value wins.
                mask = np.maximum(mask, mask_part)

        out_path = masks_dir / Path(file_name).with_suffix(".png").name

        if binary:
            # Single-channel binary mask:
            # background = 0, foreground = 255
            Image.fromarray(mask, mode="L").save(out_path)

        elif single_channel:
            # Single-channel class-ID mask:
            # background = 0, class pixels = category_id or grouped class ID
            Image.fromarray(mask, mode="L").save(out_path)

        else:
            # RGB visualization mask:
            # class IDs are converted to colors
            palette = build_palette(int(mask.max()))
            colored = colorize_mask(mask, palette)
            Image.fromarray(colored, mode="RGB").save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ARCADE COCO polygon annotations to mask PNGs"
    )

    parser.add_argument(
        "--annotations",
        "-a",
        required=True,
        help="Path to COCO JSON, e.g. train.json",
    )

    parser.add_argument(
        "--images",
        "-i",
        required=True,
        help="Path to images directory",
    )

    parser.add_argument(
        "--masks",
        "-m",
        required=True,
        help="Path to output masks directory",
    )

    parser.add_argument(
        "--groups",
        help=(
            "Semicolon-separated groups of category_ids, "
            "e.g. '0,1,2;3,4;5,6'. "
            "Each group is mapped to class IDs 1, 2, 3, ..."
        ),
    )

    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary masks with foreground=255 instead of class IDs",
    )

    parser.add_argument(
        "--single-channel",
        action="store_true",
        help=(
            "Write single-channel class-ID masks. "
            "Each pixel value is the category_id, or the grouped class ID "
            "if --groups is used. Background remains 0. "
            "No RGB colorization is applied."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.binary and args.single_channel:
        raise ValueError(
            "Use either --binary or --single-channel, not both. "
            "--binary already writes a single-channel foreground/background mask."
        )

    category_map = parse_groups(args.groups)

    write_masks(
        coco_json=Path(args.annotations),
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        binary=args.binary,
        single_channel=args.single_channel,
        category_map=category_map,
    )


if __name__ == "__main__":
    main()