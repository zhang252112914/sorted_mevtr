from config import all_args
import numpy as np
import torch
from trainer.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, args, logger):
        super(Trainer, self).__init__(args)
        self.args = args
        self.logger = logger