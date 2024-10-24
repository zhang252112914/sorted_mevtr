from config import all_args
import numpy as np
import torch
import os
import time

from trainer.base_trainer import BaseTrainer
from evaluation.evaluation import eval_epoch


class Trainer(BaseTrainer):
    def __init__(self, args, logger, model, train_dataloader, test_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, resumed_epoch=0, train_sampler=None):
        super(Trainer, self).__init__(args, resumed_epoch)
        self.args = args
        self.logger = logger
        self.train_sampler = train_sampler
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.n_gpu = n_gpu
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def train(self):
        for epoch in range(self.resumed_epoch, self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            tr_loss, global_step = self.train_epoch(epoch)
            if self.args.local_rank == 0:
                self.logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, self.args.epochs, tr_loss)

                output_model_file = self.save_model(epoch, self.args, self.model, self.optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                eval_epoch(self.args, self.model, self.test_dataloader, self.device, self.n_gpu, self.logger)

    def train_epoch(self, epoch):
        logger = self.logger
        args = self.args
        model = self.model
        train_dataloader = self.train_dataloader
        device = self.device
        n_gpu = self.n_gpu
        optimizer = self.optimizer
        scheduler = self.scheduler
        global_step = self.global_step
        local_rank = args.local_rank

        torch.cuda.empty_cache()
        model.train()
        log_step = args.n_display
        start_time = time.time()
        total_loss = 0.0
        total_loss1, total_loss2 = 0.0, 0.0

        for step, batch in enumerate(train_dataloader):
            if n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

            input_ids, input_mask, group_mask, video, video_mask, vt_mask = batch
            
            loss1, loss2, reg_loss = model(input_ids, input_mask, group_mask, video, video_mask, vt_mask)
            if args.regularize == 'none':
                loss = loss1 + args.alpha * loss2
            else:
                loss = loss1 + args.alpha * loss2 + args.reg_lambda * reg_loss
            if torch.isnan(loss):
                print(loss1, loss2, reg_loss)
                raise ValueError
            
            if args.dynamic_alpha:
                args.alpha = loss1.item() / loss2.item()

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            total_loss1 += float(loss1)
            total_loss2 += float(loss2)
            total_loss += float(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if scheduler is not None:
                    scheduler.step()  # Update learning rate schedule

                optimizer.step()
                optimizer.zero_grad()

                # https://github.com/openai/CLIP/issues/46
                if hasattr(model, 'module'):
                    torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
                else:
                    torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

                global_step += 1
                if global_step % log_step == 0 and local_rank == 0:
                    logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, %f, Reg Loss: %f, Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                len(train_dataloader),
                                "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                                float(loss1), args.alpha * float(loss2),
                                0,
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                    start_time = time.time()

        total_loss = total_loss / len(train_dataloader)
        return total_loss, global_step

    def save_model(self, epoch, args, model, optimizer, tr_loss, type_name=""):
        # Only save the model it-self
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(
            args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
        optimizer_state_file = os.path.join(
            args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
        }, optimizer_state_file)
        self.logger.info("Model saved to %s", output_model_file)
        self.logger.info("Optimizer saved to %s", optimizer_state_file)
        return output_model_file