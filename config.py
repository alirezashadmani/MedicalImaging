"""
Shared configuration for the bone segmentation project.

All default paths and parameters are centralized here.
Override via CLI arguments in each script.
"""

import os

# Default image paths (override via CLI --image argument in each script)
DEFAULT_IMAGE = os.environ.get(
    "BONE_SEG_IMAGE",
    "images/mean_image.png",
)

# Default output directory
DEFAULT_OUTPUT_DIR = os.environ.get(
    "BONE_SEG_OUTPUT",
    "output",
)


def ensure_output_dir(path=None):
    """Create the output directory if it doesn't exist."""
    out = path or DEFAULT_OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    return out
