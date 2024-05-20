import torch
from models.base import TasNetEncoder
from models.base import TasNetDecoder
from models.base import DualPathRNN
from models.audio_models import SourceSeparationModel

in_ch = 256
out_ch = 256
hid = 128
bi = True
seg = 200

e_dict = {'enc_dim': in_ch}
s_dict = {'input_size': in_ch, 'hidden_size': hid, 'output_size': out_ch, 'bidirectional': bi, 'segment_size': seg, 'num_speakers': 2}
d_dict = {'enc_dim': in_ch}

net = SourceSeparationModel(TasNetEncoder, e_dict, DualPathRNN, s_dict, TasNetDecoder, d_dict)
from datasets_ import dprnn_dataset
tmp = {
       'dataset_dir': "",
       'csv_path': "",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       }
dataset = dprnn_dataset.SegmentedSeparationDataset(
    **tmp
    )
print('TESTING')
x = torch.ones(1, 100)
out = net(x)
print(out)
print(net.eval())
print('kk')