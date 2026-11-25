"""Application package initialization."""

import os
import platform


# This must be set before PyTorch initializes MPS. Unsupported MPS operations
# can then fall back to CPU instead of aborting an inference request.
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
