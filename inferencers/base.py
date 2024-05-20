from utils.utils import setup_logger
from utils.utils import load_obj
from utils.utils import save_audio
import logging
import torch
import time
from tqdm import tqdm

class Inferencer(object):
    '''
    A base class for inference (BSS mode)
    Input
    config -- yaml config
    callable model --- model to use
    dict model_conf --- params of the passed model
    torch.nn.Dataloder --- dataloader object
    '''
    def __init__(self, config, model, model_conf, dataloader):
        logger = setup_logger('inference')
        self.logger = logging.getLogger('inference')

        self.config = config

        self.checkpoint_path = config['checkpoint_path']
        self.output_dir = config['output_dir']
        self.sample_rate = config['inference']['sample_rate']

        self.model = model(**model_conf)
        
        self.dataloader = dataloader

        self.load_model()

    def one_run(self, x):
        if not(self.device=="cpu"):
            x = load_obj(x, self.device)
        out = self.model(x)
        return [torch.squeeze(p.detach().cpu()) for p in out]
    
    def load_model(self):
        if(torch.cuda.is_available()):
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if(self.checkpoint_path is None):
            self.logger.info("WARNING! Checkpoint unspecified, initiating dry run.")
        else:
            self.logger.info(f"Using checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to(self.device)

        self.model.eval()

    def inference(self):
        self.model.eval()
        self.logger.info('Initiating inference run.')
        start_time = time.time()
        with torch.no_grad():
            ind = 0
            for mix, ref in tqdm(self.dataloader):
                if mix.dim() == 1:
                    mix = torch.unsqueeze(mix, 0)

                out = self.model(mix)
                pred = [torch.squeeze(p.detach().cpu()) for p in out]
                for i in range(len(pred)):
                    fname = self.output_dir + '/' + str(ind) + '/' + str(i) + '.wav'
                    save_audio(fname, pred[i], self.sample_rate)

        end_time = time.time()
        self.logger.info('')
        self.logger.info(f'Finished inference run in {end_time - start_time} seconds')