"""
Insurance Claim Intelligence System - Application Launcher
===========================================================
Entry point that imports and runs the Flask application from app/app.py
Automatically opens the application in your default browser.
"""

import sys
import webbrowser
import threading
from pathlib import Path
from time import sleep
from app.app import app

# Add app directory to path
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR))

def open_browser():
    """Open the Flask application in the default browser after a short delay"""
    sleep(2)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("Starting Insurance Claim Intelligence System...")
    print("Opening http://localhost:5000 in your browser...")
    
    # Start browser opener in a background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run the Flask app
    app.run(debug=True, port=5000, use_reloader=False)

