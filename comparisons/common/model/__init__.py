"""Model adapters and externally supplied licensed modules."""

import os
from pathlib import Path

external = os.environ.get("OCEAN_COMPARISON_EXTERNAL_CODE")
if external:
    __path__.append(str(Path(external) / "model"))
