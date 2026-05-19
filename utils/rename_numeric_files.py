import argparse
from pathlib import Path


def rename_numeric_files(folder: Path, target: Path, offset: int, recursive: bool) -> None:
    paths = folder.rglob("*") if recursive else folder.glob("*")
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.is_file():
            continue
        stem = path.stem
        if not stem.isdigit():
            continue
        new_stem = str(int(stem) + offset)
        rel_dir = path.parent.relative_to(folder)
        out_dir = target / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        new_path = out_dir / (new_stem + path.suffix)
        if new_path.exists():
            raise FileExistsError(f"Target exists: {new_path}")
        path.rename(new_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename numeric files by adding a fixed offset")
    parser.add_argument("--dir", "-d", required=True, help="Folder containing files to rename")
    parser.add_argument("--target", "-t", required=True, help="Target folder for renamed files")
    parser.add_argument("--offset", "-o", type=int, default=1000, help="Integer offset to add")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rename_numeric_files(Path(args.dir), Path(args.target), args.offset, args.recursive)


if __name__ == "__main__":
    main()
