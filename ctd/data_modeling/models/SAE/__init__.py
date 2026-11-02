"""SAE models, including the historical uppercase LDS import alias."""

import sys

from . import lds

# Keep old imports and pickled class paths working without tracking filenames
# that differ only by case on macOS and Windows.
LDS = lds
sys.modules[__name__ + ".LDS"] = lds
