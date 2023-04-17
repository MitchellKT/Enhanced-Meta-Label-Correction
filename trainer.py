import copy
import torch
from utils import tocuda, evaluate
import numpy as np
from meta import teacher_backward, teacher_backward_ms
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, rank, args, main_net, meta_net, enhancer, gold_loader, silver_loader, valid_loader, test_loader, num_classes, logger, exp_id=None):
        self.rank = rank
        self.args = args
        self.main_net = main_net
        self.meta_net = meta_net
        self.enhancer = enhancer
        self.gold_loader = gold_loader
        self.silver_loader = silver_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.logger = logger
        self.exp_id = exp_id

        if rank == 0:
            self.writer = SummaryWriter(args.logdir + '/' + exp_id)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self._setup_training()

    def _setup_training(self):
        ''' Fetch optimizers and schedulers for
          training the student, teacher and its enhancer'''
        args = self.args
        main_params = self.main_net.parameters() 
        meta_params = self.meta_net.parameters()
        enhancer_params = self.enhancer.parameters()

        self.main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)
        
        self.meta_opt = torch.optim.SGD(meta_params, lr=args.meta_lr,
                                    weight_decay=args.wdecay)
        self.enhancer_opt = torch.optim.SGD(enhancer_params, lr=args.meta_lr,
                                    weight_decay=args.wdecay)

        milestones = args.sched_milestones
        gamma = args.sched_gamma
        self.main_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.main_opt, milestones=milestones, gamma=gamma)
        self.meta_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.meta_opt, milestones=milestones, gamma=gamma)
        self.enhancer_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.enhancer_opt, milestones=milestones, gamma=gamma)

    def _training_iter(self, data_s, target_s, data_g, target_g):
        ''' Perform a single training iteration'''
        data_g, target_g = tocuda(self.rank, data_g), tocuda(self.rank, target_g)
        data_s, target_s = tocuda(self.rank, data_s), tocuda(self.rank, target_s)

        # bi-level optimization stage
        eta = self.main_schdlr.get_last_lr()[0]
        kwargs = {'rank': self.rank, 'args': self.args,
                'main_net': self.main_net, 'main_opt': self.main_opt,
                'teacher': self.meta_net, 'teacher_opt': self.meta_opt,
                'enhancer': self.enhancer, 'enhancer_opt': self.enhancer_opt,
                'data_s': data_s, 'target_s': target_s,
                'data_g': data_g, 'target_g': target_g,
                'eta': eta, 'num_classes': self.num_classes}
        teacher_backward_fn = teacher_backward if self.args.gradient_steps == 1 else teacher_backward_ms
        loss_g, loss_s, t_loss = teacher_backward_fn(**kwargs)

        # Update metrics
        if self.args.steps % self.args.every == 0:
            for metric, name in zip([loss_g, loss_s, t_loss], ['loss_g', 'loss_s', 't_loss']):
                dist.reduce(metric, dst=0)
                metric /= self.args.n_gpus
                if self.rank == 0:
                    self.writer.add_scalar(f'train/{name}', metric.item(), self.args.steps)

            if self.rank == 0:
                main_lr = self.main_schdlr.get_last_lr()[0]
                self.writer.add_scalar('train/main_lr', main_lr, self.args.steps)
                self.writer.add_scalar('train/gradient_steps', self.args.gradient_steps, self.args.steps)

    def train(self):
        ''' Training loop '''
        self.meta_net.train()
        self.main_net.train()

        if self.rank == 0:
            self.best_val_main = 0
        self.epoch = 0

        for epoch in range(self.args.epochs):
            self.logger.info('Epoch %d:' % epoch)
            self.args.steps = 0
            all_data_s, all_target_s = [], []
            self.epoch += 1

            if self.rank == 0:
                self.pbar = tqdm(total=len(self.silver_loader.dataset))

            for *data_s, target_s in self.silver_loader:
                
                # Setup training iteration 
                self.args.steps += 1
                if type(data_s) is list and len(data_s) == 1:
                    data_s = data_s[0]
                *data_g, target_g = next(self.gold_loader)
                if self.args.gradient_steps > 1:
                    all_data_s.append(data_s)
                    all_target_s.append(target_s)
                
                # Actual training step
                if self.args.steps % self.args.gradient_steps == 0:
                    if self.args.gradient_steps == 1:
                        self._training_iter(data_s, target_s, data_g, target_g)
                    else:
                        data_s = torch.cat(all_data_s)
                        target_s = torch.cat(all_target_s)
                        self._training_iter(data_s, target_s, data_g, target_g)
                        all_data_s, all_target_s = [], []

                # Update progress bar
                if self.rank == 0:
                    self.pbar.update(n=self.args.bs)

            # Per epoch evaluation
            if self.rank == 0:
                self.pbar.close()
                self._epoch_evaluation()
            dist.barrier()

            self.main_schdlr.step()
            self.meta_schdlr.step()
            self.enhancer_schdlr.step()

    def _epoch_evaluation(self):
        ''' Evaluate epoch in terms of main/test val/test accuracy
            Save best model if necessary '''
        val_acc_main = evaluate(self.rank, self.main_net, self.valid_loader)
        val_acc_meta = evaluate(self.rank, self.meta_net, self.valid_loader)
        test_acc_main = evaluate(self.rank, self.main_net, self.test_loader)
        test_acc_meta = evaluate(self.rank, self.meta_net, self.test_loader)

        self.logger.info('Val acc: %.4f\tTest acc: %.4f' % (val_acc_main, test_acc_main))
        self.writer.add_scalar('train/val_acc_main', val_acc_main, self.epoch)
        self.writer.add_scalar('train/test_acc_main', test_acc_meta, self.epoch)
        self.writer.add_scalar('train/val_acc_meta', val_acc_meta, self.epoch)
        self.writer.add_scalar('test/test_acc_main', test_acc_main, self.epoch)
            
        if val_acc_main > self.best_val_main:
            self.best_val_main = val_acc_main

            self.best_main_params = copy.deepcopy(self.main_net.state_dict())
            self.best_meta_params = copy.deepcopy(self.meta_net.state_dict())

            self.logger.info('Saving best models...')
            torch.save({
                'epoch': self.epoch,
                'best val main': self.best_val_main,
                'main_net': self.best_main_params,
                'meta_net': self.best_meta_params,
            }, 'models/%s_best.pth' % self.exp_id)
                
        self.writer.add_scalar('train/val_acc_best', self.best_val_main, self.epoch)
    
    def final_eval(self):
        ''' Compute final test score '''
        if self.rank == 0:   
            self.main_net.load_state_dict(self.best_main_params)
            self.meta_net.load_state_dict(self.best_meta_params)

            test_acc_main = evaluate(self.rank, self.main_net, self.test_loader)

            self.writer.add_scalar('test/acc', test_acc_main, self.args.steps)
            self.logger.info('Test acc: %.4f' % test_acc_main)
            return test_acc_main