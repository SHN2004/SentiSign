"""
SentiSign - Sign Language to Speech with Emotion.

A multi-modal system for expressive communication.
"""

import sys

# Check Python version on import
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"SentiSign requires Python 3.11 or 3.12. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Please upgrade your Python version.\n"
        f"See https://github.com/SHN2004/SentiSign#-important-python-version-requirement for details."
    )

if sys.version_info >= (3, 13):
    raise RuntimeError(
        f"SentiSign does not support Python 3.13+ yet due to PyTorch compatibility. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Please use Python 3.11 or 3.12.\n"
        f"See https://github.com/SHN2004/SentiSign#-important-python-version-requirement for details."
    )

__version__ = "0.1.0"
