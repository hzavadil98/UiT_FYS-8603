__all__ = ["AJIVE", "PCA", "data_block_heatmaps", "jive_full_estimate_heatmaps"]

import importlib
import sys

from . import jive as _jive
from . import mvdr as _mvdr
from . import ya_pca as _ya_pca

# Keep backward compatibility with existing absolute imports inside the vendored packages.
sys.modules.setdefault("jive", _jive)
sys.modules.setdefault("mvdr", _mvdr)
sys.modules.setdefault("ya_pca", _ya_pca)

PCA = importlib.import_module(".jive.PCA", __name__).PCA
block_visualization = importlib.import_module(".jive.viz.block_visualization", __name__)
data_block_heatmaps = block_visualization.data_block_heatmaps
jive_full_estimate_heatmaps = block_visualization.jive_full_estimate_heatmaps
AJIVE = importlib.import_module(".mvdr.ajive.AJIVE", __name__).AJIVE
