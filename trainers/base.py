import time
from metrics.metrics import sisnr_loss
import torch
import os
from utils.utils import setup_logger
from utils.utils import load_obj
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt


    
class Trainer(object):
    def __init__(self, config, model, train_dataloader, val_dataloader, optimizer, scheduler, gradient_clipping=5):
        logger = setup_logger('train')
        self.logger = logging.getLogger('train')
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.num_speakers = config['train']['num_speakers']
        self.num_epochs = config['train']['num_epochs']
        self.early_stop = config['train']['early_stop']
        self.checkpoint_path = config['checkpoint_path']
        self.new_checkpoint_path = config['new_checkpoint_path']
        self.plot = config['train']['plot']

        if(torch.cuda.is_available()):
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(self.device)
        self.cur_epoch = 0
        self.start = 0

        if config['checkpoint_path'] is not None:
            self.logger.info(f"Continue training from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.cur_epoch = checkpoint['epoch']
            self.start = self.cur_epoch
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.logger.info("Initiating new training run")  

        self.optimizer = optimizer
        self.scheduler = scheduler
        if gradient_clipping is not None:
            self.logger.info('Gradient clipping with maximum L2-norm of 5 will be applied.')
        self.clip = gradient_clipping

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        self.logger.info(f'Initiating train run, epoch {epoch}.')
        start_time = time.time()
        for mix, ref in tqdm(self.train_dataloader):
            mix = load_obj(mix, self.device)
            ref = load_obj(ref, self.device)

            self.optimizer.zero_grad()

            out = self.model(mix)

            l = sisnr_loss(out, ref)

            loss = l
            epoch_loss += loss.item()
            loss.backward()

            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

        end_time = time.time()
        self.logger.info('')
        self.logger.info(f'Finished training on epoch {epoch} in {end_time - start_time} seconds')
        epoch_loss = epoch_loss/self.train_dataloader.__len__()
        self.logger.info(f'Training epoch {epoch} loss: {epoch_loss}')
        return epoch_loss

    def validation(self, epoch):
        self.model.eval()
        epoch_loss = 0.0
        self.logger.info(f'Initiating validation run, epoch {epoch}.')
        start_time = time.time()
        with torch.no_grad():
            for mix, ref in tqdm(self.val_dataloader):
                mix = load_obj(mix, self.device)
                ref = load_obj(ref, self.device)

                self.optimizer.zero_grad()

                out = self.model(mix)

                l = sisnr_loss(out, ref)

                loss = l
                epoch_loss += loss.item()

        end_time = time.time()
        self.logger.info('')
        self.logger.info(f'Finished validation on epoch {epoch} in {end_time - start_time} seconds')
        epoch_loss = epoch_loss/self.val_dataloader.__len__()
        self.logger.info(f'Validation epoch {epoch} loss: {epoch_loss}')
        return epoch_loss

    def run(self):
        train_loss = []
        val_loss = []
        self.save_checkpoint(self.cur_epoch, best=False)
        v_loss = self.validation(self.cur_epoch)
        best_loss = v_loss
        no_improve = 0
        while self.cur_epoch < self.num_epochs and no_improve < self.early_stop:
            start_time = time.time()
            self.cur_epoch += 1
            self.logger.info(f'Initiating epoch {self.cur_epoch}. Loss {v_loss}')
            self.logger.info('')
            t_loss = self.train(self.cur_epoch)
            v_loss = self.validation(self.cur_epoch)

            train_loss.append(t_loss)
            val_loss.append(v_loss)

            self.scheduler.step(v_loss)

            if v_loss >= best_loss:
                no_improve += 1
                self.logger.info(
                    f'Could not find best model for {no_improve} consecutive epochs. Current best loss {best_loss}.')

            else:
                best_loss = v_loss
                no_improve = 0
                self.logger.info(
                    f'New best model found. New best loss {best_loss}.')
                self.save_checkpoint(self.cur_epoch, best=True)
                self.logger.info('Saving best model checkpoint.')
            end_time = time.time()
            self.logger.info(f'Finished epoch {self.cur_epoch}, in {end_time - start_time} seconds.')
        self.logger.info('Finished training for {self.cur_epoch} epochs.')
        self.save_checkpoint(self.cur_epoch, best=False)
        self.logger.info('Last checkpoint saved.')
        self.train_loss = train_loss
        self.val_loss = val_loss
        if self.plot is not None:
            self._plot()
    
    def _plot(self):
        self.logger.info('Plotting losses...')
        plt.title("Loss of train and test")
        x = [i for i in range(self.start, self.cur_epoch)]
        plt.plot(x, self.train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, self.val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        fname = self.config['train']['plot_path']
        if fname is not None:
            self.logger.info(f'Plotted image saved at {fname}')
            plt.savefig(self.config['train']['plot_path'])

    def save_checkpoint(self, epoch, best=True):
        '''
        save model
        best: the best model
        '''
        cpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        save_path = os.path.join(self.new_checkpoint_path,"{0}.pt".format(str(epoch)))
        if best == True:
            save_path = os.path.join(self.new_checkpoint_path,"BEST.pt")
        torch.save(cpt, save_path)