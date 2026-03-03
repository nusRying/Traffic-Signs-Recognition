#!/usr/bin/env python3
"""
run.py - Convenience wrapper for running the Traffic Sign Recognition system
"""

import subprocess
import sys


def main():
    """Run the main app with provided arguments"""
    # Pass all arguments to app.py
    result = subprocess.run([sys.executable, 'app.py'] + sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
