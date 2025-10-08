"""Run webcam emotion detection for a short window using the CNN_NeuralNet
model, then synthesize speech with Parler TTS based on the dominant emotion.

Usage example (with uv):
  uv run python webcam_emotion_tts_torch.py \
    --checkpoint resnet_emotion.pth --duration 5 --flip --imagenet-norm

Notes:
- Mirrors webcam_emotion_tts.py CLI and flow, but uses the new PyTorch CNN.
- Aggregates predictions across faces and frames during the capture window.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


# Class labels (order must match training)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# Placeholder descriptions per emotion label (edit to taste)
EMOTION_DESCRIPTIONS: Dict[str, str] = {
    "angry": (
        "An angry speaker with tense tone, firm and sharp delivery,"
        " moderate speed and slightly raised pitch; recording is clean and close."
    ),
    "disgust": (
        "A speaker with a disgusted tone, constricted articulation, and clipped phrases;"
        " moderate pace and mid pitch; high-quality close recording."
    ),
    "fear": (
        "A fearful speaker with breathy tension, cautious delivery, and uneven rhythm;"
        " moderate-slow speed and slightly higher pitch; clean recording."
    ),
    "happy": (
        "A cheerful speaker with bright tone, upbeat rhythm, and lively delivery;"
        " moderate-fast speed and slightly higher pitch; very clear close recording."
    ),
    "neutral": (
        "A neutral speaker with even tone, steady pacing, and clear articulation;"
        " moderate speed and mid pitch; high-quality near-field recording."
    ),
    "sad": (
        "A sad speaker with soft tone, gentle delivery, and slower rhythm;"
        " reduced intensity and slightly lower pitch; clean, close recording."
    ),
    "surprise": (
        "A surprised speaker with sudden emphasis, energetic articulation, and dynamic rhythm;"
        " moderate-fast speed and higher pitch; clear and close recording."
    ),
}


class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 7, linear_in_features: int = 2048):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(linear_in_features, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        return self.classifier(x)


def setup_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    return torch.device(device_arg)


def load_cnn_model(checkpoint: Path, device: torch.device, in_channels: int, linear_in_features: int) -> nn.Module:
    obj = torch.load(checkpoint, map_location=device)

    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model

    if isinstance(obj, dict) and any(k in obj for k in ["model_state", "state_dict"]):
        state_dict = obj.get("model_state", obj.get("state_dict"))
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise RuntimeError("Unsupported checkpoint format: expected full module or state dict")

    # Infer input channels from conv1 if present
    detected_in: Optional[int] = None
    for k, t in state_dict.items():
        if k.endswith("conv1.weight") and t.ndim == 4:
            detected_in = int(t.shape[1])
            break
    in_ch = detected_in if detected_in is not None else in_channels

    model = CNN_NeuralNet(in_channels=in_ch, num_classes=len(EMOTION_LABELS), linear_in_features=linear_in_features)

    # Adapt conv1 if mismatch 1<->3
    conv_key = None
    for k in ["conv1.weight", "module.conv1.weight"]:
        if k in state_dict:
            conv_key = k
            break
    if conv_key is not None:
        w = state_dict[conv_key]
        if w.ndim == 4 and w.shape[1] != model.conv1.weight.shape[1]:
            if w.shape[1] == 3 and model.conv1.weight.shape[1] == 1:
                state_dict[conv_key] = w.mean(dim=1, keepdim=True)
            elif w.shape[1] == 1 and model.conv1.weight.shape[1] == 3:
                state_dict[conv_key] = w.repeat(1, 3, 1, 1)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")

    return model.to(device).eval()


def preprocess_roi_bgr(
    roi_bgr: np.ndarray,
    img_size: int = 44,
    imagenet_norm: bool = True,
    clahe: Optional[cv2.CLAHE] = None,
    in_channels: int = 3,
) -> torch.Tensor:
    # Optional CLAHE on luminance channel
    if clahe is not None:
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        roi_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if in_channels == 1:
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        roi = roi[None, :, :]  # 1 x H x W
        if imagenet_norm:
            m, s = 0.5, 0.5
            roi = (roi - m) / s
    else:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.resize(roi_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        roi = roi_rgb.astype(np.float32) / 255.0
        roi = roi.transpose(2, 0, 1)  # 3 x H x W
        if imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            roi = (roi - mean) / std
    return torch.from_numpy(roi).unsqueeze(0)


def infer_faces(
    frame: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
    model: nn.Module,
    device: torch.device,
    img_size: int,
    imagenet_norm: bool,
    clahe: Optional[cv2.CLAHE] = None,
    in_channels: int = 3,
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    annotations: List[Tuple[Tuple[int, int, int, int], str, float]] = []
    for (x, y, w, h) in face_boxes:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        # Use the configured input channel count (matches how the model was instantiated)
        tensor = preprocess_roi_bgr(roi, img_size=img_size, imagenet_norm=imagenet_norm, clahe=clahe, in_channels=in_channels).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
        annotations.append(((x, y, w, h), EMOTION_LABELS[pred_idx.item()], float(conf.item())))
    return annotations


def draw_annotations(frame: np.ndarray, annotations: List[Tuple[Tuple[int, int, int, int], str, float]]):
    for (x, y, w, h), label, confidence in annotations:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label}: {confidence * 100:.1f}%"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top_left = (x, y - th - base)
        bottom_right = (x + tw, y)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, text, (x, y - base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def run_webcam_window(
    checkpoint: Path,
    device_str: str,
    camera_index: int,
    duration_sec: float,
    flip: bool,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    use_clahe: bool,
    clahe_clip: float,
    clahe_grid: int,
    img_size: int,
    in_channels: int,
    linear_in_features: int,
    imagenet_norm: bool,
) -> Tuple[str, Dict[str, int]]:
    device = setup_device(device_str)
    model = load_cnn_model(checkpoint, device, in_channels=in_channels, linear_in_features=linear_in_features)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    clahe = None
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(max(1, clahe_grid), max(1, clahe_grid)))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    counts: Dict[str, int] = {label: 0 for label in EMOTION_LABELS}
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Match detection call style from webcam_emotion_torch.py
            if min_size and min_size > 0:
                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size),
                )
            else:
                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                )

            annotations = infer_faces(
                frame,
                faces,
                model,
                device,
                img_size=img_size,
                imagenet_norm=imagenet_norm,
                clahe=clahe,
                in_channels=in_channels,
            )
            for _, label, _ in annotations:
                if label in counts:
                    counts[label] += 1

            draw_annotations(frame, annotations)
            cv2.imshow("Emotion Recognition (window)", frame)

            if (time.time() - start) >= duration_sec:
                break
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        # Ensure the window closes immediately at the end of the capture window
        cap.release()
        cv2.destroyAllWindows()
        # Flush GUI events so macOS/Qt backends actually close the window promptly
        for _ in range(5):
            cv2.waitKey(1)

    top_emotion = max(counts.items(), key=lambda kv: kv[1])[0] if any(v > 0 for v in counts.values()) else "neutral"
    return top_emotion, counts


def synthesize_with_parler(
    device: torch.device,
    description: str,
    prompt: str,
    output_wav: Path,
) -> Path:
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(str(output_wav), audio_arr, model.config.sampling_rate)
    return output_wav


def main():
    parser = argparse.ArgumentParser(description="Webcam emotion → Parler TTS (CNN model)")
    parser.add_argument("--checkpoint", type=Path, default=Path("resnet_emotion.pth"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=1.3)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=44, help="Input size for CNN (matches your preprocessing)")
    parser.add_argument("--linear-in-features", type=int, default=2048, help="Flattened features into final Linear")
    parser.add_argument("--imagenet-norm", action="store_true", help="Apply ImageNet normalization (recommended)")
    parser.add_argument("--output-wav", type=Path, default=Path("parler_tts_out.wav"))

    args = parser.parse_args()

    # Default TTS prompt (edit as you like)
    prompt = "Hey, how are you doing today?"

    # 1) Capture a short window and tally predictions
    top_emotion, counts = run_webcam_window(
        checkpoint=args.checkpoint,
        device_str=args.device,
        camera_index=args.camera,
        duration_sec=args.duration,
        flip=args.flip,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=args.min_size,
        use_clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        img_size=args.img_size,
        in_channels=3,
        linear_in_features=args.linear_in_features,
        imagenet_norm=args.imagenet_norm,
    )

    # Print the selected emotion immediately (before TTS) and then synthesize.
    print("\n=== Emotion Summary (last window) ===")
    for label, count in counts.items():
        if count > 0:
            print(f"  {label:>8s}: {count}")
    print(f"\nSelected emotion: {top_emotion}")

    # 2) Map emotion to a description (print it right away before TTS)
    description = EMOTION_DESCRIPTIONS.get(
        top_emotion,
        EMOTION_DESCRIPTIONS.get("neutral", "A neutral speaker with clear, near-field recording."),
    )
    print(f"Description used: {description}")

    # 3) TTS device
    tts_device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    # 4) Synthesize
    out_path = synthesize_with_parler(tts_device, description=description, prompt=prompt, output_wav=args.output_wav)

    # 5) Print TTS info
    print(f"\nParler TTS output saved to: {out_path}")


if __name__ == "__main__":
    main()
