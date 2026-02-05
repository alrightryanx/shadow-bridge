#!/usr/bin/env python3
import re
import os

BRIDGE_GUI_PATH = "shadow_bridge_gui.py"

def increment_version():
    if not os.path.exists(BRIDGE_GUI_PATH):
        print(f"Error: {BRIDGE_GUI_PATH} not found.")
        return

    with open(BRIDGE_GUI_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find APP_VERSION = "1.046"
    pattern = r'(APP_VERSION\s*=\s*")(\d+\.\d+)(")'
    match = re.search(pattern, content)

    if match:
        prefix = match.group(1)
        current_version = match.group(2)
        suffix = match.group(3)
        
        # Split into major and minor (e.g. "1" and "046")
        parts = current_version.split('.')
        
        if len(parts) == 2:
            major = parts[0]
            minor = parts[1]
            
            try:
                minor_int = int(minor)
                new_minor_int = minor_int + 1
                
                # Simple padding logic: keep at least same length
                width = len(minor)
                new_minor = f"{new_minor_int:0{width}d}"
                
                new_version = f"{major}.{new_minor}"
                
                # Replace using the full match to preserve spacing
                old_full_match = match.group(0)
                new_full_match = f'{prefix}{new_version}{suffix}'
                new_content = content.replace(old_full_match, new_full_match)
                
                with open(BRIDGE_GUI_PATH, "w", encoding="utf-8") as f:
                    f.write(new_content)
                    
                print(f"Version auto-incremented: {current_version} -> {new_version}")
            except ValueError as e:
                print(f"Error parsing minor version '{minor}': {e}")
        else:
            print(f"Warning: Version format {current_version} not standard X.YYY")
    else:
        print("Error: APP_VERSION not found in shadow_bridge_gui.py")

if __name__ == "__main__":
    increment_version()