# utils.py
import os

def ensure_directory_exists(directory):
    """Ensure a directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)