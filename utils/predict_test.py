"""
Thin wrapper that runs predict.py on the ARCADE test set.

Passes arcade/imgs/test as input and arcade/masks/predict as output
directory to the existing predict.py CLI, so all inference logic lives
in one place.

Usage:
    python utils/predict_test.py                               # uses defaults
    python utils/predict_test.py --model checkpoints/best_model.pth
    python utils/predict_test.py --model checkpoints/best_model.pth --scale 1.0
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Allow running from repo root or from utils/
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT  = REPO_ROOT / 'arcade' / 'imgs'  / 'test'
DEFAULT_OUTPUT = REPO_ROOT / 'arcade' / 'masks' / 'predict'
DEFAULT_MODEL  = REPO_ROOT / 'checkpoints' / 'best_model.pth'
PREDICT_SCRIPT = REPO_ROOT / 'predict.py'


def get_args():
    parser = argparse.ArgumentParser(
        description='Run predict.py on the ARCADE test set'
    )
    parser.add_argument('--model', '-m', type=Path, default=DEFAULT_MODEL,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', '-i', type=Path, default=DEFAULT_INPUT,
                        help='Directory of test images')
    parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT,
                        help='Directory to write predicted masks')
    parser.add_argument('--scale', '-s', type=float, default=None,
                        help='Scale factor (default: predict.py default)')
    parser.add_argument('--classes', '-c', type=int, default=None,
                        help='Number of classes (default: predict.py default)')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
    return parser.parse_args()


def main():
    args = get_args()

    # Ensure output directory exists (predict.py handles it too, but be explicit)
    args.output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PREDICT_SCRIPT),
        '--model',  str(args.model),
        '--input',  str(args.input),
        '--output', str(args.output) + '/',  # trailing slash signals directory output
        # --no-save is intentionally omitted so masks are always written
    ]

    if args.scale is not None:
        cmd += ['--scale', str(args.scale)]
    if args.classes is not None:
        cmd += ['--classes', str(args.classes)]
    if args.bilinear:
        cmd.append('--bilinear')

    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
