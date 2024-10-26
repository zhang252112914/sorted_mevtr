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
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from config import all_args
from modules.metrics import compute_metrics_together
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_factory import dataloader_factory
import modules.util_func as util_func
from trainer.trainer import Trainer
from evaluation.evaluation import eval_epoch


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
    device, n_gpu = util_func.init_device(args, logger)
    model = util_func.init_model(args, device, logger)
    model = util_func.freezze_test(model, args)

    # data
    train_dataloader, test_dataloader, train_length, \
    test_length, train_sampler = dataloader_factory(args, tokenizer, logger)

    if args.do_train:
        # training_only preparation
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        optimizer, scheduler, model = util_func.prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
            
        resumed_epoch = 0
        if args.resume_model not in [None, 'None']:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch'] + 1
            resumed_loss = checkpoint['loss']
        global_step = 0
        trainer = Trainer(args, logger, model, train_dataloader, test_dataloader, device, n_gpu, 
                          optimizer, scheduler, global_step, resumed_epoch, train_sampler)
        trainer.train()

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu, logger)

if __name__ == "__main__":
    main()