from datasets_.base import BaseDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch

class SegmentedExtractionDataset(BaseDataset):
    def __init__(self, dataset_dir, csv_path, audio_id, sample=None, num_speakers=101,
                 sample_rate=8000, segment_length=4, mode='choice', index=None, csv_cols=None, preprocessing=None):
        '''
        Parameters
        ----------
        str dataset_dir --- path to dataset root
        str csv_path --- path to dataframe with metadata
        str audio_id --- name of the id column in dataframe
        int sample --- number of (random) rows to consider (if None, all rows are considered)
        int sample_rate --- sample rate of the audio files
                            (assuming all audio share sample rate within one dataset instance)
        int segment_length --- desired length of the audio segment
        str mode --- segmenting mode (default is 'choice')
                    when set to 'choice' selects one random segment of fixed length from audio sample
                    when set to 'trim' turns audio sample into max possible segments of fixed length
        List[dict] index --- custom index (if None, the df observations are used)
        List[str] csv_cols --- list of dataframe columns to consider (if None, all columns are considered)
        dict[callable : dict] preprocessing --- dict with preprocessing functions as keys and 
                            their respective arguments as values (if None, audio data is not preprocessed)
        '''
        super(SegmentedExtractionDataset, self).__init__(dataset_dir, csv_path, index)
        df = pd.read_csv(csv_path, usecols=csv_cols)
        if sample is not None:
            df = df.sample(sample)

        self.keys = df[audio_id].tolist()
        self.mode = mode
        mix_data = dict(zip(self.keys, df[audio_id].apply(get_path, **{'data_dir': self.dataset_dir, 'folder': 'mix_clean'}).tolist()))
        df['lengths'] = df[audio_id].apply(get_path, **{'data_dir': self.dataset_dir, 'folder': 's1'})
        tgt_data = dict(zip(self.keys, df['lengths'].tolist()))
        df['lengths'] = df['lengths'].apply(self.load_audio)
        df['lengths'] = df['lengths'].apply(get_shape)
        self.preprocessing = preprocessing
        self.chunk_size = sample_rate * segment_length
        self.mix_data = []
        self.tgt_data = []
        self.aux_data = []
        self.speakers = []
        self.speaker_labels = []
        for i in tqdm(range(len(self.keys))):
            mix = self.load_audio(mix_data[self.keys[i]])
            tgt = self.load_audio(tgt_data[self.keys[i]])
            speaker = stripper(self.keys[i])
            tgt_id = self.keys[i][:self.keys[i].find('_')]
            refs = df[audio_id].loc[(df[audio_id].apply(stripper) == speaker) & (df[audio_id].apply(lambda x: x[:x.find('_')]) != tgt_id) & (df['lengths'] >= self.chunk_size)].tolist()
            if refs == []:
                continue
            chosen_aux = get_path(np.random.choice(refs), self.dataset_dir, 's1')
            aux = self.load_audio(chosen_aux)
            length = min(mix.shape[-1], tgt.shape[-1], aux.shape[-1])
            if self.mode == 'trim':
                cur_pos = 0
                while length >= self.chunk_size:
                    self.mix_data.append(mix[cur_pos:self.chunk_size + cur_pos])
                    self.tgt_data.append(tgt[cur_pos:self.chunk_size + cur_pos])
                    self.aux_data.append(aux[cur_pos:self.chunk_size + cur_pos])
                    length -= self.chunk_size
                    cur_pos += self.chunk_size
            else:
                if length >= self.chunk_size:
                    if length == self.chunk_size:
                        frame_offset = 0
                    else:
                        frame_offset = np.random.randint(0, length - self.chunk_size)
                    num_frames = self.chunk_size
                    mix = self.load_audio(
                        mix_data[self.keys[i]],
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    )
                    tgt = self.load_audio(
                        tgt_data[self.keys[i]],
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    )
                    aux = self.load_audio(
                        chosen_aux,
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    )
                    self.mix_data.append(mix)
                    self.tgt_data.append(tgt)
                    self.aux_data.append(aux)
                    ind = speaker[:speaker.find('-')]
                    if ind not in self.speaker_labels:
                        self.speaker_labels.append(ind)
                    self.speakers.append(self.speaker_labels.index(ind))

        if index is None:
            index = df.to_dict("records")
        self.n_speakers = max(num_speakers, len(self.speaker_labels))

    def __len__(self):
        return len(self.mix_data)

    def __getitem__(self, key):
        mix = self.mix_data[key]
        tgt = self.tgt_data[key]
        aux = self.aux_data[key]
        speaker_id = torch.nn.functional.one_hot(torch.tensor(self.speakers[key]), self.n_speakers).float()
        if self.preprocessing is not None:
            for f in self.preprocessing.keys():
                mix = f(mix, **self.preprocessing[f])
                tgt = f(tgt, **self.preprocessing[f])
                aux = f(aux, **self.preprocessing[f])
        
        return mix, tgt, aux, speaker_id


def get_path(s, data_dir, folder):
    x = s + '.wav'
    return os.path.join(data_dir, folder, x)

def get_shape(t):
    return t.shape[-1]

def stripper(string):
  s = string[:string.find('_')]
  return s[:s.rfind('-')]