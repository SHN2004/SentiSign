import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EMOTION_LABELS_DEFAULT = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_diseases: int = 7, linear_in_features: int = 2048):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))  # Adjusted kernel size from 4 to 2
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),  # Adjusted kernel size from 4 to 2
            nn.Flatten(),
            nn.Linear(linear_in_features, num_diseases)  # Corrected input features for the linear layer
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        return self.classifier(x)


def try_load_model(
    model_path: str,
    num_classes: int,
    in_channels: Optional[int],
    linear_in_features: int,
    device: torch.device,
) -> nn.Module:
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj
        model.eval().to(device)
        return model

    if isinstance(obj, dict) and any(k in obj for k in ["model_state", "state_dict"]):
        state_dict = obj.get("model_state", obj.get("state_dict"))
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise RuntimeError("Unsupported checkpoint format. Provide a full model or a state dict.")

    detected_in = None
    for key, tensor in state_dict.items():
        if key.endswith("conv1.weight") and tensor.ndim == 4:
            detected_in = int(tensor.shape[1])
            break
    if in_channels is None:
        in_ch = detected_in if detected_in is not None else 3
    else:
        in_ch = in_channels

    model = CNN_NeuralNet(in_channels=in_ch, num_diseases=num_classes, linear_in_features=linear_in_features)

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

    model.eval().to(device)
    return model


def preprocess_roi(
    roi_bgr: np.ndarray,
    img_size: int,
    in_channels: int,
    imagenet_norm: bool,
) -> torch.Tensor:
    roi = cv2.resize(roi_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if in_channels == 1:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = roi.astype(np.float32) / 255.0
        roi = roi[None, :, :]
        if imagenet_norm:
            m, s = 0.5, 0.5
            roi = (roi - m) / s
    else:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        roi = roi.transpose(2, 0, 1)  # 3 x H x W

        if imagenet_norm:
            mean_im = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std_im = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            roi = (roi - mean_im) / std_im

    tensor = torch.from_numpy(roi).unsqueeze(0)
    return tensor


def main():
    # Default configuration that worked well for you
    model_path = "resnet_emotion.pth"
    num_classes = 7
    in_channels: Optional[int] = 3  # use 3-channel
    linear_in_features = 2048       # trained head for 32x32/44x44 flows
    img_size = 44                   # preprocess to 44x44
    imagenet_norm = True            # apply ImageNet normalization
    camera_index = 0
    flip_frame = True
    scale_factor = 1.3
    min_neighbors = 5

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

    labels = EMOTION_LABELS_DEFAULT

    model = try_load_model(
        model_path=model_path,
        num_classes=num_classes,
        in_channels=in_channels,
        linear_in_features=linear_in_features,
        device=device,
    )

    try:
        eff_in_ch = int(model.conv1.in_channels)  # type: ignore[attr-defined]
    except Exception:
        eff_in_ch = (1 if in_channels == 1 else 3)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try --camera 0/1 or check permissions.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if flip_frame:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

            for (x, y, w, h) in faces:
                roi_bgr = frame[y:y+h, x:x+w]
                tensor = preprocess_roi(
                    roi_bgr,
                    img_size=img_size,
                    in_channels=eff_in_ch,
                    imagenet_norm=imagenet_norm,
                )
                tensor = tensor.to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                label = labels[idx] if idx < len(labels) else str(idx)
                conf = float(probs[idx])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}: {conf*100:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

            cv2.imshow("Emotion Detection (PyTorch)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
