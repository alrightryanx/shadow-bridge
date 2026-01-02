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
import json
import argparse
import os

# Add shadow-bridge to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using local Stable Diffusion via SSH"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate an image')
    gen_parser.add_argument('prompt', help='Text prompt for image generation')
    gen_parser.add_argument('--model', default='sd-xl-turbo',
                           choices=['sd-1.5', 'sd-xl', 'sd-xl-turbo'],
                           help='Model to use (default: sd-xl-turbo)')
    gen_parser.add_argument('--steps', type=int, default=4,
                           help='Number of inference steps (default: 4 for turbo, 20 for others)')
    gen_parser.add_argument('--width', type=int, default=1024,
                           help='Image width (default: 1024)')
    gen_parser.add_argument('--height', type=int, default=1024,
                           help='Image height (default: 1024)')
    gen_parser.add_argument('--seed', type=int, default=None,
                           help='Seed for reproducibility')
    gen_parser.add_argument('--guidance', type=float, default=0.0,
                           help='Guidance scale (default: 0.0 for turbo, 7.5 for others)')
    gen_parser.add_argument('--negative', type=str, default=None,
                           help='Negative prompt')

    # Status command
    subparsers.add_parser('status', help='Check image service status')

    # Setup command - ensures model is downloaded
    setup_parser = subparsers.add_parser('setup', help='Ensure model is ready')
    setup_parser.add_argument('--model', default='sd-xl-turbo',
                             help='Model to prepare')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'generate':
            # Adjust defaults based on model
            steps = args.steps
            guidance = args.guidance

            if args.model == 'sd-xl-turbo':
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
                prompt=args.prompt,
                negative_prompt=args.negative,
                model=args.model,
                width=args.width,
                height=args.height,
                steps=steps,
                guidance_scale=guidance,
                seed=args.seed
            )
            print(json.dumps(result))

        elif args.command == 'status':
            from web.services.image_service import get_image_service_status, get_image_setup_status
            status = get_image_service_status()
            setup = get_image_setup_status()
            print(json.dumps({
                "service": status,
                "setup": setup
            }))

        elif args.command == 'setup':
            from web.services.image_service import get_image_generation_service
            service = get_image_generation_service()
            try:
                model = service.warmup_model(args.model)
                print(json.dumps({
                    "success": True,
                    "model": model,
                    "message": f"Model {model} is ready"
                }))
            except Exception as e:
                print(json.dumps({
                    "success": False,
                    "error": str(e)
                }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
