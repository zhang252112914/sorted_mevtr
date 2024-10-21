from config import all_args
import numpy as np
import torch

class Trainer():
    def __init(self, args, logger):
        self.args = args
        self.logger = logger

    def init_device(self):
        args = self.args
        logger = self.logger
        local_rank = args.local_rank

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
        n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
        args.n_gpu = n_gpu
        if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
            raise ValueError(
                "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                    args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
        return device, n_gpu