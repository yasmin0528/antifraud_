"""
AMLSIM data preprocessing — one-shot pipeline for all variants.

Usage:
    python feature_engineering/amlsim_preprocess.py --variant HI-Small
    python feature_engineering/amlsim_preprocess.py --variant LI-Small --force
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from datasets.amlsim import preprocess_amlsim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess AMLSIM dataset")
    parser.add_argument("--variant", default="HI-Small",
                        choices=["HI-Small", "LI-Small",
                                 "HI-Medium", "LI-Medium",
                                 "HI-Large", "LI-Large"])
    parser.add_argument("--data-dir", default="data/AMLSIM",
                        help="AMLSIM data root directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-run preprocessing even if artifacts exist")
    args = parser.parse_args()

    data_root = os.path.join(_PROJECT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    trans_path = os.path.join(data_root, f"{args.variant}_Trans.csv")
    accounts_path = os.path.join(data_root, f"{args.variant}_accounts.csv")
    output_dir = os.path.join(data_root, f"{args.variant}_processed")

    for path in (trans_path, accounts_path):
        if not os.path.exists(path):
            logger.error("Missing required file: %s", path)
            sys.exit(1)

    logger.info("Preprocessing %s", args.variant)
    logger.info("  Transactions: %s", trans_path)
    logger.info("  Accounts:     %s", accounts_path)
    logger.info("  Output:       %s", output_dir)

    artifacts = preprocess_amlsim(
        trans_path=trans_path,
        accounts_path=accounts_path,
        output_dir=output_dir,
        force=args.force,
    )

    logger.info("Done. Artifacts:")
    for key, path in artifacts.items():
        size = os.path.getsize(path) / (1024**3)
        logger.info("  %s: %s (%.2f GB)", key, path, size)


if __name__ == "__main__":
    main()
