"""
RVC Voice Model Training Script
Standalone script for training voice models from audio samples
Can be called from ShadowBridge or run independently
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train RVC voice model from audio samples"
    )

    parser.add_argument("--name", required=True, help="Name for the voice model")

    parser.add_argument(
        "--description", default="", help="Description of the voice model"
    )

    parser.add_argument(
        "--voice-type",
        choices=["singing", "speech"],
        default="singing",
        help="Type of voice model (singing or speech)",
    )

    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="Paths to audio sample files (minimum 3 recommended)",
    )

    parser.add_argument(
        "--gender",
        choices=["male", "female", "neutral", None],
        default=None,
        help="Gender of the voice",
    )

    parser.add_argument(
        "--age-group",
        choices=["child", "young", "adult", "senior", None],
        default=None,
        help="Age group of the voice",
    )

    parser.add_argument(
        "--accent",
        default=None,
        help="Accent of the voice (e.g., 'American', 'British')",
    )

    parser.add_argument(
        "--output-dir", default=None, help="Output directory for model files"
    )

    parser.add_argument("--config", help="Path to RVC configuration JSON file")

    args = parser.parse_args()

    # Validate samples
    if len(args.samples) < 1:
        logger.error("At least 1 training sample required (3+ recommended)")
        return 1

    # Check if samples exist
    valid_samples = []
    for sample in args.samples:
        path = Path(sample)
        if path.exists():
            valid_samples.append(str(path))
        else:
            logger.warning(f"Sample not found: {sample}")

    if len(valid_samples) == 0:
        logger.error("No valid samples found")
        return 1

    logger.info(f"Training voice model: {args.name}")
    logger.info(f"Type: {args.voice_type}")
    logger.info(f"Samples: {len(valid_samples)}")
    logger.info(f"Gender: {args.gender or 'Not specified'}")
    logger.info(f"Age group: {args.age_group or 'Not specified'}")
    logger.info(f"Accent: {args.accent or 'Not specified'}")

    # Load RVC manager
    try:
        from web.utils.rvc_manager import RVCManager

        rvc_path = None
        models_path = args.output_dir

        if args.config:
            with open(args.config, "r") as f:
                config = json.load(f)
                rvc_path = config.get("rvc_webui_dir")
                models_path = config.get("models_dir")

        rvc = RVCManager(rvc_path=rvc_path, models_path=models_path)

        # Train model
        result = await rvc.train_voice_model(
            name=args.name,
            description=args.description,
            voice_type=args.voice_type,
            training_samples=valid_samples,
            gender=args.gender,
            age_group=args.age_group,
            accent=args.accent,
        )

        if result.get("success"):
            logger.info("=" * 50)
            logger.info("  TRAINING SUCCESSFUL")
            logger.info("=" * 50)
            logger.info(f"Model ID: {result['model_id']}")
            logger.info(f"Model path: {result['model_path']}")
            logger.info(f"Metadata path: {result['metadata_path']}")
            logger.info(f"Training time: {result['training_time']:.2f} seconds")
            logger.info(f"Sample count: {result['quality_metrics']['sample_count']}")
            logger.info(f"Quality: {result['quality_metrics']['quality']}")
            logger.info(
                f"Estimated accuracy: {result['quality_metrics']['estimated_accuracy']:.2f}"
            )
            logger.info("=" * 50)
            return 0
        else:
            logger.error("=" * 50)
            logger.error("  TRAINING FAILED")
            logger.error("=" * 50)
            logger.error(f"Error: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
