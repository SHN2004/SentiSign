# SentiSign Setup Guide

Complete installation and setup instructions for SentiSign.

## Prerequisites

### System Requirements

**Minimum**:
- Operating System: Windows 10, macOS 10.15+, or Linux (Ubuntu 20.04+)
- Python: 3.11 or 3.12 (NOT 3.13+)
- RAM: 8 GB
- Webcam: 720p, 30 FPS
- Disk Space: 3 GB (includes models and dependencies)

**Recommended**:
- Python: 3.11 (most tested and stable)
- RAM: 16 GB
- GPU: NVIDIA with CUDA 11.8+ (6+ GB VRAM)
- Webcam: 1080p, 60 FPS

### Software Prerequisites

1. **Python 3.11 or 3.12**

   This project requires Python 3.11 or 3.12. **Python 3.13+ will NOT work** due to PyTorch compatibility.

   ```bash
   # Check your Python version
   python --version

   # If you see "Python 3.13.x", you need to install Python 3.11
   ```

   **Installing Python 3.11**:

   **macOS** (using Homebrew):
   ```bash
   brew install python@3.11
   ```

   **Windows**:
   - Download from: https://www.python.org/downloads/release/python-31110/
   - During installation, check "Add Python to PATH"

   **Linux** (Ubuntu/Debian):
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```

   **Verify installation**:
   ```bash
   python3.11 --version  # Should show 3.11.x
   ```

2. **uv** (Python package manager)
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or on Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Verify installation
   uv --version
   ```

3. **Git**
   ```bash
   git --version
   ```

4. **(Optional) CUDA Toolkit** (for NVIDIA GPU support)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Version: 11.8 or higher recommended

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/SHN2004/SentiSign.git
cd SentiSign
```

### Step 2: Create Virtual Environment

Using **uv** (recommended):

```bash
# Create virtual environment
# uv will automatically use Python 3.11 from .python-version file
uv venv

# Or explicitly specify Python 3.11:
uv venv --python 3.11

# Or if you have python3.11 in PATH:
uv venv --python python3.11

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

**Note**: The repository includes a `.python-version` file that tells `uv` to automatically use Python 3.11. If you don't have Python 3.11 installed, `uv` will show an error message.

### Step 3: Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .
```

This will install:
- PyTorch (2.8.0+)
- OpenCV (4.12.0+)
- Parler TTS
- Transformers
- And all other dependencies

### Step 4: Verify Installation

Run the verification script:

```bash
python -c "import torch; import cv2; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.8.0 (or higher)
OpenCV: 4.12.0.88 (or higher)
CUDA available: True (if GPU is available, False otherwise)
```

### Step 5: Download Models (Already Included)

The emotion detection model is already included in the repository at:
```
models/emotion/resnet_emotion.pth
```

On first run of TTS demos, Parler TTS model (~1 GB) will be automatically downloaded from Hugging Face.

## Platform-Specific Setup

### macOS (Apple Silicon)

PyTorch with MPS (Metal Performance Shaders) support:

```bash
# MPS should work automatically
# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Note**: For TTS on M1/M2 Macs, CPU mode is recommended due to memory limitations with large models.

### Windows

1. Install Visual C++ Redistributable (required for PyTorch)
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Camera permissions:
   - Settings → Privacy → Camera → Allow apps to access camera

### Linux

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-pip libopencv-dev
   ```

2. Camera permissions:
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Log out and log back in
   ```

## Running Your First Demo

### Basic Emotion Detection

```bash
python demos/run_emotion_webcam.py
```

You should see:
1. Webcam window opens
2. Face detection with green bounding box
3. Emotion label and confidence displayed
4. Press 'q' to quit

### Emotion + TTS Demo

```bash
python demos/run_emotion_tts.py --duration 5
```

You should see:
1. Webcam captures for 5 seconds
2. Emotion summary printed
3. TTS model generates audio (first run downloads model)
4. Audio saved to `parler_tts_out.wav`

## Troubleshooting

### Issue: "Cannot open camera"

**Solution**:
1. Check camera permissions in system settings
2. Try different camera index:
   ```bash
   python demos/run_emotion_webcam.py --camera 1
   ```
3. Close other apps using the camera (Zoom, Teams, etc.)

### Issue: "CUDA out of memory"

**Solution**:
1. Use CPU mode:
   ```bash
   python demos/run_emotion_webcam.py --device cpu
   ```
2. Reduce batch size (for future modules)
3. Close other GPU-intensive applications

### Issue: "Failed to load Haar cascade"

**Solution**:
1. Reinstall OpenCV:
   ```bash
   uv pip install --force-reinstall opencv-python
   ```

### Issue: TTS download fails

**Solution**:
1. Check internet connection
2. Set Hugging Face cache directory:
   ```bash
   export HF_HOME=/path/to/cache  # macOS/Linux
   set HF_HOME=C:\path\to\cache   # Windows
   ```
3. Manual download:
   ```python
   from transformers import AutoTokenizer
   from parler_tts import ParlerTTSForConditionalGeneration

   model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
   tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
   ```

### Issue: Poor emotion detection accuracy

**Solution**:
1. Improve lighting conditions
2. Try CLAHE preprocessing:
   ```bash
   python demos/run_emotion_webcam.py --clahe
   ```
3. Adjust face detection parameters:
   ```bash
   python demos/run_emotion_webcam.py --scale-factor 1.1 --min-neighbors 3
   ```

## Performance Optimization

### GPU Acceleration

1. Verify CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
   ```

2. Install CUDA-enabled PyTorch (if not auto-detected):
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Webcam Performance

1. Reduce resolution (if laggy):
   - Adjust in your OS camera settings

2. Lower detection frequency:
   - Modify code to skip frames (e.g., process every 2nd frame)

## Environment Variables

Optional configuration:

```bash
# Hugging Face cache directory
export HF_HOME=/path/to/cache

# OpenCV camera backend (Linux)
export OPENCV_VIDEOIO_PRIORITY_LIST=V4L2

# PyTorch CUDA settings
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
```

## Updating

To update to the latest version:

```bash
cd SentiSign
git pull origin main
uv pip install -e . --upgrade
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf SentiSign
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
- See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
- Check [MODULE_GUIDE.md](MODULE_GUIDE.md) to add new features

## Getting Help

- Open an issue on GitHub
- Check existing issues for similar problems
- Include error messages and system info when reporting bugs
