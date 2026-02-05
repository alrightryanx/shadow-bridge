"""
RVC Audio Conversion Script
Standalone script for converting audio using RVC voice models
Can be called from ShadowBridge or run independently
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))


async def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(
        description="Convert audio to target voice using RVC model"
    )

    parser.add_argument(
        "--input", required=True, help="Path to input audio file to convert"
    )

    parser.add_argument(
        "--model", required=True, help="Voice model ID to use for conversion"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output path for converted audio (default: input_file_converted.wav)",
    )

    parser.add_argument(
        "--no-pitch-correction",
        action="store_true",
        help="Disable automatic pitch correction",
    )

    parser.add_argument("--config", help="Path to RVC configuration JSON file")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_{args.model}.wav"

    logger.info(f"Converting audio: {input_path}")
    logger.info(f"Using voice model: {args.model}")
    logger.info(f"Pitch correction: {not args.no_pitch_correction}")
    logger.info(f"Output: {output_path}")

    # Load RVC manager
    try:
        from web.utils.rvc_manager import RVCManager

        rvc_path = None
        models_path = None

        if args.config:
            import json

            with open(args.config, "r") as f:
                config = json.load(f)
                rvc_path = config.get("rvc_webui_dir")
                models_path = config.get("models_dir")

        rvc = RVCManager(rvc_path=rvc_path, models_path=models_path)

        # Convert audio
        result = await rvc.convert_audio(
            input_audio=str(input_path),
            voice_model_id=args.model,
            pitch_correction=not args.no_pitch_correction,
        )

        if result.get("success"):
            logger.info("=" * 50)
            logger.info("  CONVERSION SUCCESSFUL")
            logger.info("=" * 50)
            logger.info(f"Output path: {result['output_path']}")
            logger.info(f"Duration: {result.get('duration', 'N/A')} seconds")
            logger.info(f"Conversion time: {result['conversion_time']:.2f} seconds")
            logger.info(f"Model used: {result['model_used']}")
            logger.info("=" * 50)

            # Copy to final output location if different
            result_output_path = Path(result["output_path"])
            if result_output_path != output_path:
                import shutil

                shutil.copy2(result_output_path, output_path)
                logger.info(f"Copied to: {output_path}")

            return 0
        else:
            logger.error("=" * 50)
            logger.error("  CONVERSION FAILED")
            logger.error("=" * 50)
            logger.error(f"Error: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
