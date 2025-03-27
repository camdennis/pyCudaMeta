"""
Dynamic loader for libpyCudaMeta.so
"""
import sys
import os

# Adds a relative path to find libpyCudaElasto
sys.path.append(os.path.join(
    os.path.dirname(__file__), "..", "build"
))

import libpyCudaMeta
