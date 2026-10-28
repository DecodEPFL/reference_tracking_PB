import torch, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

if torch.cuda.is_available():
    print('[INFO] GPU available')
else:
    print('[INFO] running on CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The step-by-step closed-loop rollouts are made of many tiny ops: extra intra-op threads only
# spin (measured on an M3 Max: 1 thread 4.5 s/epoch and 4.5 CPU-s, 10 threads 5.1 s/epoch and 52 CPU-s,
# identical losses). Override with the env var TORCH_NUM_THREADS.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 1)))
