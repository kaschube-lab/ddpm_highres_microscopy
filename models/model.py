import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from data.dataset import get_min_max, denorm_tensor
from copy import deepcopy

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.lr_scheduler = CosineAnnealingLR(self.optG, T_max=20)
        self.schedulers.append(self.lr_scheduler)
        self.resume_training()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

        self.loss_on_img = self.loss_fn.__name__ in ['fft_loss', 'fft_mse_loss', 'edge_loss', 'charbonnier_edge_loss', 'lpips_loss']

        self.phase = self.opt['phase']
        self.dataset_name = self.opt['datasets'][self.phase]['which_dataset']['args']['type']
        self.min_max = get_min_max()
        self.denorm_fn = denorm_tensor()
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'): # TODO: adjust 
        dict = {
            'gt_image': self.denorm_fn(self.gt_image.detach()[:].float().cpu()),
            'cond_image': self.denorm_fn(self.cond_image.detach()[:].float().cpu()),
        }
        if phase != 'train':
            dict.update({
                'output': self.denorm_fn(self.output.detach()[:].float().cpu()),
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            # ret_path.append('GT_{}'.format(self.path[idx]))
            # ret_result.append(self.gt_image[idx].detach().float().cpu())

            # ret_path.append('Process_{}'.format(self.path[idx]))
            # ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            if self.loss_on_img:
                # create frozen net for the image-based loss
                frozen_net = deepcopy(self.netG)
                frozen_net.requires_grad_(False)
                frozen_net.eval()
            else:
                frozen_net = None
            loss = self.netG(
                self.gt_image, self.cond_image, loss_on_img=self.loss_on_img, 
                frozen_net=frozen_net, min_max=self.min_max)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            # wandb.log({
            #     "iter": self.iter, 
            #     "epoch": self.epoch, 
            #     "train_loss": loss.item()
            #     })
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
        # wandb.finish()

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    self.output, self.visuals = self.netG.module.restoration(
                        self.cond_image, sample_num=self.sample_num, min_max=self.min_max)
                else:
                    self.output, self.visuals = self.netG.restoration(
                        self.cond_image, sample_num=self.sample_num, min_max=self.min_max)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                # if self.opt['save_images']:
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    self.output, self.visuals = self.netG.module.restoration(
                        self.cond_image, sample_num=self.sample_num, min_max=self.min_max)
                else:
                    self.output, self.visuals = self.netG.restoration(
                        self.cond_image, sample_num=self.sample_num, min_max=self.min_max)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
