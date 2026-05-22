import argparse
import logging
import os
import random
import sys
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from hyperparameters import (
    DEFAULT_AMP,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BILINEAR,
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_CLASSES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOAD,
    DEFAULT_SCALE,
    DEFAULT_SAVE_CHECKPOINT,
)

dir_img = Path('./arcade/imgs/train/')
dir_val = Path('./arcade/imgs/val/')
dir_train_mask = Path('./arcade/masks/train/')
dir_val_mask = Path('./arcade/masks/val/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = DEFAULT_SAVE_CHECKPOINT,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        class_weights: Optional[Union[str, List[float]]] = None,
):
    # 1. Create dataset (fallback to BasicDataset if Carvana naming does not match)
    dataset = BasicDataset(dir_img, dir_train_mask, img_scale)

    # 2. Load a fixed validation set
    train_set = dataset
    val_set = train_set
    # BasicDataset(dir_val, dir_val_mask, img_scale)

    n_train = len(train_set)
    n_val = len(val_set)

    # 2b. Resolve class weights once per run
    resolved_class_weights = resolve_class_weights(class_weights, train_set, model.n_classes, device)

    # 3. Create data loaders
    # DataLoader settings tuned for GPU training
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # W&B can be disabled via WANDB_DISABLED=true
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
           dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
               save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
               class_weights=class_weights)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, loss, LR scheduler, and AMP scaler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # Reduce LR when validation Dice plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=999999)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Use Dice + weighted CE/BCE for more stable optimization
    if model.n_classes == 1:
        pos_weight = None
        if resolved_class_weights is not None:
            if resolved_class_weights.numel() != 2:
                raise ValueError('Binary segmentation expects 2 class weights: background,foreground')
            pos_weight = (resolved_class_weights[1] / resolved_class_weights[0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        if resolved_class_weights is not None and resolved_class_weights.numel() != model.n_classes:
            raise ValueError(f'Expected {model.n_classes} class weights, got {resolved_class_weights.numel()}')
        criterion = nn.CrossEntropyLoss(weight=resolved_class_weights)
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Forward pass in mixed precision if enabled
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # Binary segmentation: BCE + Dice over the single channel
                        loss = criterion(masks_pred.squeeze(1), true_masks.float()) + dice_loss(
                            F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False
                        )
                    else:
                        # Multiclass segmentation: CE + Dice over one-hot masks
                        loss = criterion(masks_pred, true_masks) + dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # Backprop with optional gradient scaling for AMP
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                # Unscale before clipping so thresholds apply to true gradients
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # Validate a few times per epoch for faster feedback
                # division_step = (n_train // (5 * batch_size))
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info('Validation Dice score: {}'.format(val_score))
        
        # Save a checkpoint per epoch for resuming or inference
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        elif epoch == epochs:
            logging.info('Final epoch reached, saving final model...')
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'final_model.pth'))
            logging.info('Final model saved!')
            
        
        

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=DEFAULT_LOAD, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=DEFAULT_SCALE, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=DEFAULT_AMP, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=DEFAULT_BILINEAR, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=DEFAULT_CLASSES, help='Number of classes')
    parser.add_argument('--class-weights', type=parse_class_weights_arg, default=DEFAULT_CLASS_WEIGHTS,
                        help='Comma-separated weights per class or "auto"')

    return parser.parse_args()


def parse_class_weights_arg(value: str) -> Union[str, List[float]]:
    value = value.strip()
    if value.lower() == 'auto':
        return 'auto'
    try:
        return [float(v) for v in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'Class weights must be "auto" or comma-separated numbers, e.g. "1,2,3".'
        ) from exc


def resolve_class_weights(
        class_weights: Optional[Union[str, List[float]]],
        train_set,
        n_classes: int,
        device: torch.device,
) -> Optional[torch.Tensor]:
    if class_weights is None:
        return None

    weight_classes = 2 if n_classes == 1 else n_classes
    if class_weights == 'auto':
        weights = compute_class_weights(train_set, weight_classes)
    else:
        weights = torch.tensor(class_weights, dtype=torch.float32)
        if weights.numel() != weight_classes:
            raise ValueError(f'Expected {weight_classes} class weights, got {weights.numel()}')

    logging.info('Class weights: %s', weights.tolist())
    return weights.to(device)


def compute_class_weights(train_set, n_classes: int) -> torch.Tensor:
    # Use a single-threaded loader to avoid overhead during weighting
    weight_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for batch in tqdm(weight_loader, desc='Computing class weights', unit='img'):
        mask = batch['mask'].view(-1)
        counts += torch.bincount(mask, minlength=n_classes).double()

    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights.float()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            class_weights=args.class_weights
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            class_weights=args.class_weights
        )
