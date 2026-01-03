#!/usr/bin/env python3
"""
CLI for image generation via SSH.

Usage:
    python shadow_image_cli.py generate "A sunset over mountains"
    python shadow_image_cli.py generate "A cat" --model sd-xl-turbo --steps 4
    python shadow_image_cli.py status

Returns JSON output for easy parsing by Android app.
"""

import sys
import json as json_module
import argparse
import os
import base64

# Add shadow-bridge to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using local Stable Diffusion via SSH"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an image")
    gen_parser.add_argument(
        "prompt",
        nargs="?",
        help="Text prompt for image generation (optional if --prompt_b64 is used)",
    )
    gen_parser.add_argument("--prompt_b64", help="Base64 encoded text prompt (safer)")
    gen_parser.add_argument(
        "--model",
        default="sd-xl-turbo",
        choices=["sd-1.5", "sd-xl", "sd-xl-turbo"],
        help="Model to use (default: sd-xl-turbo)",
    )
    gen_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4 for turbo, 20 for others)",
    )
    gen_parser.add_argument(
        "--width", type=int, default=1024, help="Image width (default: 1024)"
    )
    gen_parser.add_argument(
        "--height", type=int, default=1024, help="Image height (default: 1024)"
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )
    gen_parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="Guidance scale (default: 0.0 for turbo, 7.5 for others)",
    )
    gen_parser.add_argument(
        "--negative", type=str, default=None, help="Negative prompt"
    )

    # Inpaint command
    inpaint_parser = subparsers.add_parser("inpaint", help="Inpaint an image")
    inpaint_parser.add_argument(
        "--image_b64",
        help="Base64 encoded source image (mutually exclusive with --stdin)",
    )
    inpaint_parser.add_argument(
        "--mask_b64", help="Base64 encoded mask image (mutually exclusive with --stdin)"
    )
    inpaint_parser.add_argument(
        "--prompt_b64",
        help="Base64 encoded text prompt (mutually exclusive with --stdin)",
    )
    inpaint_parser.add_argument(
        "--negative_b64",
        help="Base64 encoded negative prompt (mutually exclusive with --stdin)",
    )
    inpaint_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin to avoid command line length limits",
    )

    # Remove Background command
    rembg_parser = subparsers.add_parser(
        "remove-background", help="Remove background from image"
    )
    rembg_parser.add_argument(
        "--image_b64",
        help="Base64 encoded source image (mutually exclusive with --stdin)",
    )
    rembg_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin to avoid command line length limits",
    )

    # Status command
    subparsers.add_parser("status", help="Check image service status")

    # Setup command - ensures model is downloaded
    setup_parser = subparsers.add_parser("setup", help="Ensure model is ready")
    setup_parser.add_argument("--model", default="sd-xl-turbo", help="Model to prepare")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    def print_json(data):
        """Print JSON with strict markers to avoid parsing issues."""
        print("<<<JSON_START>>>")
        print(json_module.dumps(data))
        print("<<<JSON_END>>>")

    try:
        if args.command == "generate":
            # Handle prompt input (prefer base64)
            prompt = args.prompt
            if args.prompt_b64:
                prompt = base64.b64decode(args.prompt_b64).decode("utf-8")

            if not prompt:
                print_json({"success": False, "error": "No prompt provided"})
                sys.exit(1)

            # Adjust defaults based on model
            steps = args.steps
            guidance = args.guidance

            if args.model == "sd-xl-turbo":
                # Turbo model works best with fewer steps and no guidance
                if steps == 4:  # Default
                    steps = 4
                if guidance == 0.0:  # Default
                    guidance = 0.0
            else:
                # Non-turbo models need more steps
                if steps == 4:  # Default was for turbo
                    steps = 20
                if guidance == 0.0:  # Default was for turbo
                    guidance = 7.5

            from web.services.image_service import get_image_generation_service

            service = get_image_generation_service()
            result = service.generate_image(
                prompt=prompt,
                negative_prompt=args.negative,
                model=args.model,
                width=args.width,
                height=args.height,
                steps=steps,
                guidance_scale=guidance,
                seed=args.seed,
                source="cli",
            )
            print_json(result)

        elif args.command == "inpaint":
            from web.services.image_service import get_image_generation_service

            service = get_image_generation_service()

            if args.stdin:
                # Read input from stdin to avoid command line length limits
                try:
                    input_data = json_module.load(sys.stdin)
                    image_b64 = input_data.get("image_b64")
                    mask_b64 = input_data.get("mask_b64")
                    prompt_b64 = input_data.get("prompt_b64")
                    negative_b64 = input_data.get("negative_b64")

                    if not all([image_b64, mask_b64, prompt_b64]):
                        raise ValueError("Missing required fields in stdin JSON")

                    prompt = base64.b64decode(prompt_b64).decode("utf-8")
                    negative = (
                        base64.b64decode(negative_b64).decode("utf-8")
                        if negative_b64
                        else None
                    )

                except Exception as e:
                    print_json(
                        {
                            "success": False,
                            "error": f"Failed to read stdin input: {str(e)}",
                        }
                    )
                    sys.exit(1)
            else:
                # Use command line arguments (legacy, will fail for large images)
                if not all([args.image_b64, args.mask_b64, args.prompt_b64]):
                    print_json(
                        {
                            "success": False,
                            "error": "Missing required arguments: --image_b64, --mask_b64, --prompt_b64",
                        }
                    )
                    sys.exit(1)

                prompt = base64.b64decode(args.prompt_b64).decode("utf-8")
                negative = (
                    base64.b64decode(args.negative_b64).decode("utf-8")
                    if args.negative_b64
                    else None
                )
                image_b64 = args.image_b64
                mask_b64 = args.mask_b64

            # Use the correct method name: inpaint_image (not inpaint)
            result = service.inpaint_image(
                image_base64=image_b64,
                mask_base64=mask_b64,
                prompt=prompt,
                negative_prompt=negative or "",
            )
            print_json(result)

        elif args.command == "remove-background":
            from web.services.image_service import get_bg_removal_service

            service = get_bg_removal_service()

            if args.stdin:
                # Read input from stdin to avoid command line length limits
                try:
                    input_data = json_module.load(sys.stdin)
                    image_b64 = input_data.get("image_b64")

                    if not image_b64:
                        raise ValueError(
                            "Missing required field 'image_b64' in stdin JSON"
                        )

                except Exception as e:
                    print_json(
                        {
                            "success": False,
                            "error": f"Failed to read stdin input: {str(e)}",
                        }
                    )
                    sys.exit(1)
            else:
                # Use command line arguments (legacy, will fail for large images)
                if not args.image_b64:
                    print_json(
                        {
                            "success": False,
                            "error": "Missing required argument: --image_b64",
                        }
                    )
                    sys.exit(1)

                image_b64 = args.image_b64

            # Use the background removal service
            result = service.remove_background(image_base64=image_b64)
            print_json(result)

        elif args.command == "status":
            from web.services.image_service import (
                get_image_service_status,
                get_image_setup_status,
            )

            status = get_image_service_status()
            setup = get_image_setup_status()
            print_json({"service": status, "setup": setup})

        elif args.command == "setup":
            from web.services.image_service import get_image_generation_service

            service = get_image_generation_service()
            try:
                model = service.warmup_model(args.model)
                print_json(
                    {
                        "success": True,
                        "model": model,
                        "message": f"Model {model} is ready",
                    }
                )
            except Exception as e:
                print_json({"success": False, "error": str(e)})

    except Exception as e:
        print_json({"success": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
