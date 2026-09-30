"""
Streamlit Community Cloud Root Entrypoint
Automatically detected by share.streamlit.io upon repository connection.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import and execute the UI application
from ui.streamlit_app import *
