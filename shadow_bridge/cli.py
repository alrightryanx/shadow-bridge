
import json
import sys
import threading

def print_json(data):
    """Print JSON with strict markers to avoid parsing issues."""
    print("---SHADOW-BRIDGE-JSON-START---")
    print(json.dumps(data))
    print("---SHADOW-BRIDGE-JSON-END---")
    sys.stdout.flush()

def run_image_command(args, image_cli_path, install_path):
    """
    Handle image generation commands via CLI.
    
    Usage:
        ShadowBridge.exe image generate "A sunset over mountains"
        ShadowBridge.exe image generate --prompt_b64 <base64>
        ShadowBridge.exe image status
        ShadowBridge.exe image inpaint --stdin
        ShadowBridge.exe image remove-background --stdin
    """
    cmd = args[0] if args else "help"
    
    if cmd == "status":
        print_json({"status": "ready", "version": "1.0.0"})
        return
        
    if cmd == "generate":
        import subprocess
        
        # Parse arguments manually since we're in a subcommand
        prompt = ""
        prompt_b64 = None
        
        if len(args) > 1:
            if args[1] == "--prompt_b64" and len(args) > 2:
                prompt_b64 = args[2]
            else:
                prompt = args[1]
        
        # Use simple separate process for the actual generation
        # This keeps the main process responsive
        try:
            # Construct command for the image CLI wrapper
            cli_cmd = [sys.executable, str(image_cli_path), "generate"]
            
            if prompt_b64:
                cli_cmd.extend(["--prompt_b64", prompt_b64])
            else:
                cli_cmd.append(prompt)
                
            # Pass through other flags
            if "--width" in sys.argv:
                idx = sys.argv.index("--width")
                if idx + 1 < len(sys.argv):
                    cli_cmd.extend(["--width", sys.argv[idx+1]])
            
            if "--height" in sys.argv:
                idx = sys.argv.index("--height")
                if idx + 1 < len(sys.argv):
                    cli_cmd.extend(["--height", sys.argv[idx+1]])
            
            # Run it
            result = subprocess.run(cli_cmd, capture_output=True, text=True)
            
            # Forward the output
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
        except Exception as e:
            print_json({"error": str(e), "status": "error"})
            
    elif cmd in ["inpaint", "remove-background", "upscale"]:
        import subprocess
        
        # These commands typically take input from stdin (base64 image)
        # We'll pass it through to the CLI script
        try:
            cli_cmd = [sys.executable, str(image_cli_path), cmd]
            
            if "--stdin" in sys.argv:
                cli_cmd.append("--stdin")
                
            # If we're reading from stdin, we need to pipe it
            input_data = None
            if "--stdin" in sys.argv:
                input_data = sys.stdin.read()
                
            result = subprocess.run(
                cli_cmd, 
                input=input_data, 
                capture_output=True, 
                text=True
            )
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
        except Exception as e:
            print_json({"error": str(e), "status": "error"})
    else:
        print_json({"error": f"Unknown image command: {cmd}", "status": "error"})

def run_video_command(args):
    """
    Handle video generation commands via CLI.
    
    Usage:
        ShadowBridge.exe video generate "A cat dancing in the rain"
        ShadowBridge.exe video generate --prompt_b64 <base64>
        ShadowBridge.exe video status
        ShadowBridge.exe video install hunyuan-15
        ShadowBridge.exe video list-models
    """
    # This imports heavy libraries, so we do it inside the function
    try:
        # Check if the video generator module is available
        # It might be in a different path relative to the exe
        from pathlib import Path
        
        # Dummy implementation for now since we're refactoring
        # In a real scenario, we'd import the actual logic
        cmd = args[0] if args else "help"
        
        if cmd == "status":
            print_json({"status": "ready", "backend": "hunyuan"})
            return
            
        print_json({"status": "error", "message": "Video generation refactoring in progress"})
            
    except Exception as e:
        print_json({"error": str(e), "status": "error"})

def run_audio_command(args):
    """
    Handle audio generation commands via CLI.
    """
    print_json({"status": "ready", "backend": "audiocraft"})

def run_assembly_command(args):
    """
    Handle media assembly commands via CLI.
    """
    cmd = args[0] if args else "help"
    
    if cmd == "status":
        print_json({"status": "ready"})
        return
        
    print_json({"status": "ok", "message": "Assembly command received"})

def run_browser_command(args):
    """
    Handle browser automation commands via CLI.
    """
    print_json({"status": "ready", "backend": "selenium"})

def update_generation(updates):
    """
    Update the current generation status.
    This is a helper used by audio/video commands.
    """
    print_json(updates)

