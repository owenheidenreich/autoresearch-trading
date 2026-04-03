"""Dataset construction pipeline.

Downloads market data, computes features and oracle labels, produces
v2/data.pt. See docs/v2/data_contract.md for the dataset format.

v1 prepare.py was 178KB and mixed data download, feature computation,
label generation, and normalization. v2 splits these:
- v2/pipeline/build_dataset.py (this file): orchestration and I/O
- v2/pipeline/sources/: data source adapters (IBKR, Polygon)
- v2/core/features.py: feature computation
- v2/core/labels.py: oracle label computation

v1 origin: training/prepare.py (download_*, cache management, make_dataloader)
"""
# TODO: DatasetBuilder class
# TODO: CLI (--skip-download, --use-spx, --quick, --tier=1|2|3)
# TODO: Incremental update (append new dates, re-split)
# TODO: Fingerprinting (raw data hash, feature version, label version)
