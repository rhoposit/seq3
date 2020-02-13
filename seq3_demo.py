import os, sys
import torch
from generate.utils import compress_seq3_single
from sys_config import BASE_DIR

src_file = sys.argv[1]
out_file = sys.argv[2]


checkpoint = "seq3"
seed = 1
device = "cuda"
verbose = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

results = compress_seq3_single(checkpoint, src_file, out_file, device, mode="attention")
for item in results:
    print(item[0])
