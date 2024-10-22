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
from dataloaders.data_factory import dataloader_factory
import modules.util_func as util_func



torch.distributed.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=86400))
global logger

@record
def main():
    # environment
    global logger
    args = all_args.get_args()
    args, logger = all_args.set_seed_logger(args)
    assert args.task_type == "retrieval"

    # components
    tokenizer = ClipTokenizer()
    device, n_gpu = util_func.init_device(args, args.local_rank)
    model = util_func.init_model(args, device)
    model = util_func.freezze_test(model, args)

    # data
    train_dataloader, test_dataloader, train_length, \
    test_length, train_sampler = dataloader_factory(args, tokenizer, logger)

    if args.do_train:
        # training_only preparation
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        optimizer, scheduler, model = util_func.prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)



if __name__ == "__main__":
    main()