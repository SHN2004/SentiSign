"""
Phase 1: Offline Feature Extraction

Extracts ResNet-18 features from PHOENIX-2014-T video frames and saves them to disk.
The CNN is frozen and used only for feature extraction (eval mode, no gradients).

Input: Raw PNG frames from PHOENIX-2014-T/features/fullFrame-210x260px/{train,dev,test}/
Output: .pt files with shape (T, 512) per video
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_feature_extractor(device: torch.device) -> nn.Module:
    """
    Load ResNet-18 pretrained on ImageNet and remove the final classification layer.
    Returns a model that outputs (B, 512) features.
    """
    # Load pretrained ResNet-18
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Remove the final fully connected layer
    # ResNet-18 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
    # We want features after avgpool, which gives (B, 512, 1, 1)
    feature_extractor = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,
        nn.Flatten()  # (B, 512, 1, 1) -> (B, 512)
    )

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor


def get_transform() -> transforms.Compose:
    """
    Get the preprocessing transform for frames.
    Resize to 224x224 and normalize with ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def extract_features_for_video(
    frame_dir: Path,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Extract features for all frames in a video directory.

    Args:
        frame_dir: Path to directory containing PNG frames
        model: Feature extraction model
        transform: Image preprocessing transform
        device: Device to run on
        batch_size: Number of frames to process at once

    Returns:
        Tensor of shape (T, 512) where T is number of frames
    """
    # Get sorted list of frame files
    frame_files = sorted(frame_dir.glob("*.png"))

    if len(frame_files) == 0:
        raise ValueError(f"No PNG frames found in {frame_dir}")

    all_features = []

    # Process frames in batches
    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i + batch_size]

        # Load and transform frames
        batch_tensors = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            tensor = transform(img)
            batch_tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)

        # Extract features (no gradients needed)
        with torch.no_grad():
            features = model(batch)  # (B, 512)

        all_features.append(features.cpu())

    # Concatenate all batches
    return torch.cat(all_features, dim=0)  # (T, 512)


def process_split(
    phoenix_root: Path,
    split: str,
    output_root: Path,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32
) -> int:
    """
    Process all videos in a split (train/dev/test).

    Args:
        phoenix_root: Root path to PHOENIX-2014-T dataset
        split: One of 'train', 'dev', 'test'
        output_root: Where to save extracted features
        model: Feature extraction model
        transform: Image preprocessing transform
        device: Device to run on
        batch_size: Batch size for frame processing

    Returns:
        Number of videos processed
    """
    # Input: PHOENIX-2014-T/features/fullFrame-210x260px/{split}/
    input_dir = phoenix_root / "features" / "fullFrame-210x260px" / split

    # Output: output_root/{split}/
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Get all video directories
    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    print(f"\nProcessing {split} split: {len(video_dirs)} videos")

    processed = 0
    for video_dir in tqdm(video_dirs, desc=f"Extracting {split}"):
        video_id = video_dir.name
        output_path = output_dir / f"{video_id}.pt"

        # Skip if already processed
        if output_path.exists():
            processed += 1
            continue

        try:
            features = extract_features_for_video(
                video_dir, model, transform, device, batch_size
            )
            torch.save(features, output_path)
            processed += 1
        except Exception as e:
            print(f"\nError processing {video_id}: {e}")

    return processed


def extract_all_features(
    phoenix_root: str,
    output_root: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None
) -> dict:
    """
    Extract ResNet-18 features for all videos in PHOENIX-2014-T dataset.

    Args:
        phoenix_root: Path to PHOENIX-2014-T directory
        output_root: Where to save features (default: phoenix_root/features/resnet18)
        batch_size: Number of frames to process at once
        device: Device to use ('cuda' or 'cpu', auto-detected if None)

    Returns:
        Dictionary with counts of processed videos per split
    """
    phoenix_root = Path(phoenix_root)

    if output_root is None:
        output_root = phoenix_root / "features" / "resnet18"
    else:
        output_root = Path(output_root)

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Phoenix root: {phoenix_root}")
    print(f"Output root: {output_root}")

    # Load model and transform
    print("\nLoading ResNet-18 feature extractor...")
    model = get_feature_extractor(device)
    transform = get_transform()

    # Process each split
    results = {}
    for split in ["train", "dev", "test"]:
        try:
            count = process_split(
                phoenix_root, split, output_root,
                model, transform, device, batch_size
            )
            results[split] = count
        except ValueError as e:
            print(f"Skipping {split}: {e}")
            results[split] = 0

    print("\n" + "=" * 50)
    print("Feature extraction complete!")
    print(f"  Train: {results.get('train', 0)} videos")
    print(f"  Dev:   {results.get('dev', 0)} videos")
    print(f"  Test:  {results.get('test', 0)} videos")
    print("=" * 50)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ResNet-18 features from PHOENIX-2014-T frames"
    )
    parser.add_argument(
        "phoenix_root",
        help="Path to PHOENIX-2014-T directory"
    )
    parser.add_argument(
        "--output",
        help="Output directory (default: phoenix_root/features/resnet18)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (auto-detected if not specified)"
    )

    args = parser.parse_args()

    extract_all_features(
        phoenix_root=args.phoenix_root,
        output_root=args.output,
        batch_size=args.batch_size,
        device=args.device
    )
