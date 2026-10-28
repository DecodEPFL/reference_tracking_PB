import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed.*` imports
sys.path.insert(1, os.path.join(BASE_DIR, 'experiments', 'robots'))
