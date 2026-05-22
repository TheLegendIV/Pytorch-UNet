#!/usr/bin/env python
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hyperparameters as h
from utils import coco2png, rgb2gs


def clear_dir(target: Path) -> None:
    if not str(target):
        raise ValueError("Refusing to clear empty path")
    target.mkdir(parents=True, exist_ok=True)
    for entry in target.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(out_path))


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def main() -> None:
    train_src = resolve_repo_path(h.dir_train_imgs_src)
    train_json = resolve_repo_path(h.dir_train_annotations)
    val_src = resolve_repo_path(h.dir_val_imgs_src)
    val_json = resolve_repo_path(h.dir_val_annotations)

    train_img = resolve_repo_path(h.dir_img)
    val_img = resolve_repo_path(h.dir_val)
    train_mask = resolve_repo_path(h.dir_train_mask)
    val_mask = resolve_repo_path(h.dir_val_mask)
    preview_color = resolve_repo_path(h.dir_preview_colorized)
    preview_raw = resolve_repo_path(h.dir_preview_raw)
    groupings = h.groupings

    clear_dir(train_mask)
    clear_dir(val_mask)
    clear_dir(train_img)
    clear_dir(val_img)
    clear_dir(preview_color)
    clear_dir(preview_raw)

    copy_tree(train_src, train_img)
    copy_tree(val_src, val_img)

    rgb2gs.convert_directory(train_img, train_img, recursive=True)
    rgb2gs.convert_directory(val_img, val_img, recursive=True)

    category_map = coco2png.parse_groups(groupings)
    coco2png.write_masks(train_json, train_img, train_mask, False, category_map)
    coco2png.write_masks(val_json, val_img, val_mask, False, category_map)




if __name__ == "__main__":
    main()
