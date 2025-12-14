# SentiSign

**Sign Language to Speech with Emotion**

SentiSign is a multi-modal system that translates sign language to speech while incorporating emotional context for natural, expressive communication.

## Project Vision

```
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│ Webcam  │───▶│ Sign Language│───▶│  Context-   │───▶│Expressive│───▶ 🔊
│         │    │ Recognition  │    │  Aware      │    │   TTS    │
│         │    │ (Transformer)│    │Integration  │    │          │
└─────────┘    └──────────────┘    │   (SLM)     │    └──────────┘
     │                             └─────────────┘
     │         ┌──────────────┐            ▲
     └────────▶│   Emotion    │────────────┘
               │  Detection   │
               │  (ResNet)    │
               └──────────────┘
```

## Current Status

- ✅ **Emotion Detection** - Real-time facial emotion recognition (7 emotions)
- ✅ **Expressive TTS** - Emotion-aware speech synthesis using Parler TTS
- 🚧 **Sign Language Recognition** - In development (Transformer-based)
- 📋 **Context Integration** - Planned (SLM-based)
- 📋 **Full Pipeline** - Planned

## Quick Start

### ⚠️ Important: Python Version Requirement

**This project requires Python 3.11 or 3.12.**

Python 3.13+ is **NOT supported** due to PyTorch compatibility issues. If you encounter a "DLL initialization routine failed" error, you're likely using Python 3.13.

Check your Python version:
```bash
python --version  # Should show 3.11.x or 3.12.x
```

If you have Python 3.13, install Python 3.11 and recreate your virtual environment. See [docs/SETUP.md](docs/SETUP.md) for detailed installation instructions.

### Prerequisites

- Python 3.11 or 3.12 (NOT 3.13+)
- Webcam
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SHN2004/SentiSign.git
cd SentiSign
```

2. Create virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

### Running Demos

#### Emotion Detection Demo

```bash
python demos/run_emotion_webcam.py
```

Press 'q' to quit. The demo shows real-time emotion detection with bounding boxes and confidence scores.

#### Emotion + TTS Demo

```bash
python demos/run_emotion_tts.py --duration 5 --text "Hello, how are you doing?"
```

This demo:
1. Captures your emotion for 5 seconds
2. Determines the dominant emotion
3. Synthesizes speech with emotion-appropriate voice

#### Advanced Options

```bash
# Use specific camera
python demos/run_emotion_webcam.py --camera 1

# Disable mirror mode
python demos/run_emotion_webcam.py --no-flip

# Use CLAHE preprocessing for better contrast
python demos/run_emotion_webcam.py --clahe

# Longer capture for TTS
python demos/run_emotion_tts.py --duration 10

# Custom TTS output
python demos/run_emotion_tts.py --text "Your custom message" --output my_audio.wav
```

## Project Structure

```
SentiSign/
├── sentisign/              # Main source code package
│   ├── emotion/            # Emotion detection module
│   ├── sign_language/      # Sign language recognition (in progress)
│   ├── tts/                # Text-to-speech module
│   ├── integration/        # Context-aware integration (planned)
│   └── utils/              # Shared utilities
├── models/                 # Pre-trained model weights
│   └── emotion/            # Emotion detection models
├── demos/                  # Standalone demonstration scripts
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## Modules

### Emotion Detection

7-class facial emotion recognition using ResNet-inspired CNN:
- Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Real-time webcam processing
- 44×44 input resolution
- See [sentisign/emotion/README.md](sentisign/emotion/README.md)

### Expressive TTS

Emotion-aware speech synthesis using Parler TTS:
- Voice characteristics adapt to detected emotion
- 7 preset emotion voice profiles
- Custom voice description support
- See [sentisign/tts/README.md](sentisign/tts/README.md)

### Sign Language Recognition (Coming Soon)

Transformer-based sign language gesture recognition.
- See [sentisign/sign_language/README.md](sentisign/sign_language/README.md)

### Context Integration (Planned)

SLM-based integration of sign language and emotion.
- See [sentisign/integration/README.md](sentisign/integration/README.md)

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System architecture and data flow
- [Setup Guide](docs/SETUP.md) - Detailed installation and environment setup
- [Contributing](docs/CONTRIBUTING.md) - Contribution guidelines and workflow
- [Module Guide](docs/MODULE_GUIDE.md) - How to add new modules

## Technology Stack

- **Deep Learning**: PyTorch 2.0+
- **Computer Vision**: OpenCV
- **TTS**: Parler TTS (Hugging Face Transformers)
- **Package Management**: uv
- **Language**: Python 3.11 or 3.12

## Hardware Support

- **CUDA** (NVIDIA GPUs) - Full support, recommended for production
- **MPS** (Apple Silicon) - Supported for emotion detection
- **CPU** - Supported but slower

## Team

Built with collaboration and passion by the SentiSign team.

## License

[Add license information]

## Acknowledgments

- Parler TTS model from Hugging Face
- OpenCV for computer vision utilities
- PyTorch for deep learning framework

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is under active development. Features marked as "Coming Soon" or "Planned" are being worked on.
