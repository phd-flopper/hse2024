from models.audio_models import SourceSeparationModel
from datasets_.dprnn_dataset import SourceSeparationDataset
from datasets_.dprnn_dataset import SegmentedSeparationDataset
from datasets_.base import get_dataloader
from models.base import TasNetEncoder
from models.base import TasNetDecoder
from models.base import DualPathRNN
from utils.utils import save_audio
import torchaudio
from inferencers import Inferencer
from omegaconf import OmegaConf
conf = OmegaConf.load('')
in_ch = 256
out_ch = 64
hid = 128
bi = True
seg = 200
e_dict = {'enc_dim': in_ch}
s_dict = {'input_size': in_ch, 'hidden_size': hid, 'output_size': out_ch, 'bidirectional': bi, 'segment_size': seg}
d_dict = {'enc_dim': in_ch}
model_conf = {
    'encoder': TasNetEncoder, 
    'e_params': e_dict,
    'separator': DualPathRNN,
    's_params': s_dict, 
    'decoder': TasNetDecoder, 
    'd_params': d_dict
    }
conf1 = OmegaConf.load('')
tmp1 = {
       'dataset_dir': "",
       'csv_path': "",
       'mix_data': "mixture_path",
       'ref_data': ["source_1_path", "source_2_path"],
       'audio_id': "mixture_ID",
       'preprocessing': None
       }
test_dataloader = get_dataloader(SourceSeparationDataset, tmp1, is_train=False)

audio_tensor, sample_rate = torchaudio.load('.wav')
z = audio_tensor
print(z)
print(test_dataloader.dataset.__len__())
inferencer = Inferencer(conf, SourceSeparationModel, model_conf, test_dataloader)
#x = inferencer.one_run([[m, r] for m, r in test_dataloader][0][0])
x = inferencer.one_run(z)
print(x)
inferencer1 = Inferencer(conf1, SourceSeparationModel, model_conf, test_dataloader)
#y = inferencer1.one_run([[m, r] for m, r in test_dataloader][0][0])
y = inferencer1.one_run(z)
print(y)
for i in range(2):
    fname = '' + '/' + 'spk_trained' + '/' + str(i) + '.wav'
    x[i] = x[i].unsqueeze(0)
    save_audio(fname, x[i], 8000)
for i in range(2):
    fname = '' + '/' + 'spk_dry' + '/' + str(i) + '.wav'
    y[i] = y[i].unsqueeze(0)
    save_audio(fname, y[i], 8000)