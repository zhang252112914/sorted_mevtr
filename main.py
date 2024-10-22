from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.distributed.elastic.multiprocessing.errors import record
import numpy as np
import random
import os
import datetime
import time
import argparse

from config import all_args
from modules.metrics import compute_metrics_together
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
import modules.init_func as init_func



torch.distributed.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=86400))
global logger

@record
def main():
    # basic preparation
    global logger
    args = all_args.get_args()
    args, logger = all_args.set_seed_logger(args)
    
    tokenizer = ClipTokenizer()
    device, n_gpu = init_func.init_device(args, args.local_rank)
    model = init_func.init_model(args, device)

if __name__ == "__main__":
    main()