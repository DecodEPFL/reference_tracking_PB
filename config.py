import torch, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

# device is resolved at import time, before argparse runs, so read --no-gpu from sys.argv directly
if '--no-gpu' in sys.argv:
    print('[INFO] --no-gpu set, running on CPU')
    device = torch.device("cpu")
elif torch.cuda.is_available():
    print('[INFO] GPU available')
    device = torch.device("cuda:0")
else:
    print('[INFO] running on CPU')
    device = torch.device("cpu")
