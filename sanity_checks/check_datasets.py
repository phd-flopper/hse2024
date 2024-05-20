from datasets_.dprnn_dataset import SourceSeparationDataset
from datasets_.dprnn_dataset import SegmentedSeparationDataset
from datasets_.extraction_dataset import SegmentedExtractionDataset
from datasets_.base import get_dataloader
import logging
from utils.utils import setup_logger

def identity_transform(audio, a, b=0, c=None):
    '''
    Returns the audio as passed. For testing purposes.
    Parameters
    ----------
    torch.Tensor audio --- audio tensor to process
    '''
    return audio


tmp = {
       'dataset_dir': "",
       'csv_path': "mixture_train_mix_both.csv",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       'preprocessing': {identity_transform: {'a': None, 'b': None}}
       }
dataset = SourceSeparationDataset(
    **tmp
    )

tmp1 = {
       'dataset_dir': "",
       'csv_path': "\mixture_train_mix_both.csv",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       'preprocessing': {identity_transform: {'a': None, 'b': None}}
       }
dataset1 = SegmentedSeparationDataset(
    **tmp
    )

#torchaudio.save('\tmp.wav', torch.unsqueeze(dataset.__getitem__(0)['mix'][:32000], 0), 8000)
#writes audio
print(dataset.keys)

logger = setup_logger('dataset')
logger = logging.getLogger('dataset')
print('WARWAR')
print(dataset.__getitem__(0))
print('WARWAR')
print(dataset.__len__())
print(len(dataset.mix_data))
print(dataset.keys)
print()
print(dataset.mix_data)
print()
print(dataset.ref_data)
print(dataset.__getitem__(0))
logger.info('testing done')
print(dataset.__len__())
print(dataset1.__len__())

from itertools import permutations


print('AAAAAAAAAAAAAAAAAAAAAAAAA')
print(dataset1.__getitem__(0)[1][0])
print(dataset1.__len__())
x = dataset1.__getitem__(0)[1]
y = x = dataset1.__getitem__(0)[1]

tmp = {
       'dataset_dir': "Libri2Mix\wav8k\min\dev",
       'csv_path': "LibriMix\metadata\Libri2Mix\libri2mix_dev-clean.csv",
       'audio_id': "mixture_ID",
       'sample': 50,
       'preprocessing': None
       }
dataset = SegmentedExtractionDataset(
    **tmp
    )
x = dataset.__getitem__(0)
print(x)
import torchaudio
import torch
for i in x:
    torchaudio.save(f'Code\{i}.wav', torch.unsqueeze(i, 0), 8000)