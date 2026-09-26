"""Load licensed upstream cores from the verified external cache."""

import os
from pathlib import Path

external = os.environ.get("OCEAN_COMPARISON_EXTERNAL_CODE")
if external:
    __path__.append(str(Path(external) / "vendor"))
