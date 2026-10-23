"""Safe gallery paths and atomic persistence shared by the recognizers."""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def gallery_directory(root: Path | str, name: str) -> Path:
    """Return a single portable directory name confined to the gallery root."""
    if (not isinstance(name, str) or not name.strip() or name != name.strip()
            or name.startswith(".") or name.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError("Use a plain person/pet name without path separators or reserved characters")
    base = Path(root).resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError("Gallery directory must stay inside its root")
    return target


def save_gallery(path: Path, data: dict[str, np.ndarray]) -> None:
    """Replace an NPZ only after the complete new archive has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".gallery-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            with zipfile.ZipFile(stream, "w") as archive:
                for name, value in data.items():
                    gallery_directory(path.parent, name)
                    with archive.open(f"{name}.npy", "w") as array_file:
                        np.lib.format.write_array(array_file, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
