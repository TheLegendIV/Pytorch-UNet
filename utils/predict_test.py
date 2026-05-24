"""
Batch prediction script for the ARCADE test set.

Reads images from arcade/imgs/test, runs inference with a saved checkpoint,
and writes predicted masks (as uint8 PNGs, pixel values = class indices) to
arcade/masks/predict.

Usage:
    python utils/predict_test.py                              # uses defaults
    python utils/predict_test.py --model checkpoints/best_model.pth
    python utils/predict_test.py --model checkpoints/best_model.pth --scale 1.0
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Allow running from repo root or from utils/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unet import UNet
from utils.data_loading import BasicDataset
from hyperparameters import (
    DEFAULT_CLASSES,
    DEFAULT_PREDICT_SCALE,
    DEFAULT_BILINEAR,
)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

DEFAULT_INPUT  = REPO_ROOT / 'arcade' / 'imgs'  / 'test'
DEFAULT_OUTPUT = REPO_ROOT / 'arcade' / 'masks' / 'predict'
DEFAULT_MODEL  = REPO_ROOT / 'checkpoints' / 'best_model.pth'


def predict_image(net, pil_img, mask_values, scale, device):
    """Run inference on a single PIL image; return (H, W) uint8 numpy array."""
    net.eval()
    tensor = torch.from_numpy(
        BasicDataset.preprocess(mask_values, pil_img, scale, is_mask=False)
    ).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        logits = net(tensor)
        # Upsample back to original resolution
        logits = F.interpolate(logits, (pil_img.size[1], pil_img.size[0]), mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks for the ARCADE test set')
    parser.add_argument('--model', '-m', type=Path, default=DEFAULT_MODEL,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', '-i', type=Path, default=DEFAULT_INPUT,
                        help='Directory of test images')
    parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT,
                        help='Directory to write predicted masks')
    parser.add_argument('--scale', '-s', type=float, default=DEFAULT_PREDICT_SCALE,
                        help='Input image scale factor')
    parser.add_argument('--classes', '-c', type=int, default=DEFAULT_CLASSES,
                        help='Number of segmentation classes')
    parser.add_argument('--bilinear', action='store_true', default=DEFAULT_BILINEAR,
                        help='Use bilinear upsampling in UNet decoder')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Collect input images
    img_files = sorted(p for p in args.input.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not img_files:
        raise FileNotFoundError(f'No images found in {args.input}')
    logging.info(f'Found {len(img_files)} test images in {args.input}')

    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logging.info(f'Saving predicted masks to {args.output}')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load model
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', list(range(args.classes)))
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    logging.info(f'Loaded checkpoint: {args.model}')

    # Predict and save
    for img_path in tqdm(img_files, desc='Predicting', unit='img'):
        pil_img  = Image.open(img_path)
        pred     = predict_image(net, pil_img, mask_values, args.scale, device)
        out_path = args.output / (img_path.stem + '.png')
        Image.fromarray(pred).save(out_path)

    logging.info(f'Done. {len(img_files)} masks saved to {args.output}')


if __name__ == '__main__':
    main()
