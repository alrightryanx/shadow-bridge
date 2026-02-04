from web.app import create_app
import os
import sys

if __name__ == "__main__":
    # Add current directory to path so 'web' module is found
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting minimal ShadowBridge API server...")
    app = create_app()
    app.run(host="127.0.0.1", port=6767, debug=False)
