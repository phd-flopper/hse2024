import OmegaConf
from models.base import TasNetEncoder
from models.base import DualPathRNN
from models.base import TasNetDecoder
from models.audio_models import SourceSeparationModel
from datasets_.base import get_dataloader
from datasets_.dprnn_dataset import SourceSeparationDataset, SegmentedSeparationDataset
from datasets_.extraction_dataset import SegmentedExtractionDataset
from models.audio_models import SpEx_PlusDPRNN
from trainers.base import Trainer
from torch.optim import ReduceLROnPlateau
from time import time
from trainers.base import Trainer

import torch

conf = OmegaConf.load('config.yaml')

in_ch = 256
out_ch = 64
hid = 128
bi = True
seg = 200
e_dict = {'enc_dim': in_ch}
s_dict = {'input_size': in_ch, 'hidden_size': hid, 'output_size': out_ch, 'bidirectional': bi, 'segment_size': seg}
d_dict = {'enc_dim': in_ch}
tmp = {
       'dataset_dir': "",
       'csv_path': ".csv",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       'preprocessing': None,
       'sample': 200
       }
tmp1 = {
       'dataset_dir': "",
       'csv_path': ".csv",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       'preprocessing': None,
       'sample': 50
       }

model = SourceSeparationModel(TasNetEncoder, e_dict, DualPathRNN, s_dict, TasNetDecoder, d_dict)
print('model built')
print("gpu is available:", torch.cuda.is_available())

tmp2 = {
        'num_speakers': 2,
        'num_epochs': 2,
        'early_stop': 10,
        'checkpoint_path': 'BEST.pt',
        'new_checkpoint_path': ''
        }

optimizer = torch.optim.Adam(model.parameters())
print('optimizer built')
train_dataloader = get_dataloader(SegmentedSeparationDataset, tmp, batch_size=1)
print('train_dataloader built', train_dataloader.__len__())
val_dataloader = get_dataloader(SegmentedSeparationDataset, tmp1, batch_size=1, is_train=False)
print('val_dataloader built', val_dataloader.__len__())
scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        verbose=True)
print('scheduler built')
trainer = Trainer(conf, model, train_dataloader, val_dataloader, optimizer, scheduler)
print('trainer built')
start_time = time.time()
trainer.run()
end_time = time.time()
print(f'finished run, took {end_time - start_time} seconds')



clip_norm=None
min_lr=0
patience=0
factor=0.5
logging_period=100
no_impr=6
batch_size=1
segment_size=4
epochs = 1
batch_size = 16
num_workers = 4
L = 0.0025
N = 256
B = 8
O = 256
P = 512
Q = 3
spk_embed_dim = 256
causal = False
sample_rate = 8000
chunk_size = 4
lr = 1e3
tmp = {
      'dataset_dir': "Libri2Mix/wav8k/min/train-100",
       'csv_path': ".csv",
       'audio_id': "mixture_ID",
       'sample': 5000,
       'preprocessing': None
       }
from omegaconf import OmegaConf
conf = OmegaConf.load('check.yaml')
train_dataloader = get_dataloader(SegmentedExtractionDataset, tmp, batch_size=1)
num_speaks = train_dataloader.dataset.n_speakers
tmp1 = {
    'dataset_dir': "Libri2Mix/wav8k/min/dev",
    'csv_path': "libri2mix_dev-clean.csv",
    'audio_id': "mixture_ID",
    'sample': 1000,
    'preprocessing': None,
    'num_speakers': num_speaks
    }
print('train_dataloader built', train_dataloader.__len__())
val_dataloader = get_dataloader(SegmentedExtractionDataset, tmp1, batch_size=1, is_train=False)
print('val_dataloader built', val_dataloader.__len__())
L = int(L * sample_rate)
model1 = SpEx_PlusDPRNN(L=L, N=N, B=B, O=O, P=P, Q=Q, num_spks=num_speaks, spk_embed_dim=spk_embed_dim, causal=causal)
print('model1 built')
lr=0.01
weight_decay=1e-5
optimizer = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=weight_decay)
sample_rate = 8000
scheduler = ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=factor, 
        patience=patience, 
        min_lr=min_lr, 
        verbose=True)
trainer = TSETrainer(conf, model1, train_dataloader, val_dataloader, optimizer, scheduler)
print('trainer built')
start_time = time.time()
trainer.run()
end_time = time.time()
print(f'finished run, took {end_time - start_time} seconds')