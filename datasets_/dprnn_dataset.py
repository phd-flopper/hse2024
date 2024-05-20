from datasets_.base import BaseDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class SourceSeparationDataset(BaseDataset):
    def __init__(self, dataset_dir, csv_path, mix_data, ref_data, 
                 audio_id, index=None, csv_cols=None, preprocessing=None):
        '''
        Parameters
        ----------
        str dataset_dir --- path to dataset root
        str csv_path --- path to dataframe with metadata
        str mix_data --- relative path to audio mix / name of a corresponding column in dataframe
        List[str] ref_data --- relative paths to (ground truth) separate audio sources / 
                                                            names of corresponding columns in dataframe
        str audio_id --- name of the id column in dataframe
        List[dict] index --- custom index (if None, the df observations are used)
        List[str] csv_cols --- list of dataframe columns to consider (if None, all columns are considered)
        dict[callable : dict] preprocessing --- dict with preprocessing functions as keys and 
                            their respective arguments as values (if None, audio data is not preprocessed)
        '''

        df = pd.read_csv(csv_path, usecols=csv_cols)

        self.keys = df[audio_id].tolist()
        self.mix_data = dict(zip(self.keys, df[mix_data].tolist()))
        self.ref_data = [dict(zip(self.keys, df[ref].tolist())) for ref in ref_data]
        self.preprocessing = preprocessing

        if index is None:
            index = df.to_dict("records")

        super(SourceSeparationDataset, self).__init__(dataset_dir, csv_path, index)

    def __len__(self):
        return len(self.mix_data)

    def __getitem__(self, key):
        ind = self.keys[key]
        mix = self.load_audio(os.path.join(os.path.split(self.dataset_dir)[0], self.mix_data[ind]))
        ref = [self.load_audio(os.path.join(os.path.split(self.dataset_dir)[0], ref[ind])) for ref in self.ref_data]
        if self.preprocessing is not None:
            for f in self.preprocessing.keys():
                mix = f(mix, **self.preprocessing[f])
                ref = [f(r, **self.preprocessing[f]) for r in ref]
        
        return mix, ref


class SegmentedSeparationDataset(BaseDataset):
    def __init__(self, dataset_dir, csv_path, mix_data, ref_data, audio_id, sample=None,
                 sample_rate=8000, segment_length=4, mode='choice', index=None, csv_cols=None, preprocessing=None):
        '''
        Parameters
        ----------
        str dataset_dir --- path to dataset root
        str csv_path --- path to dataframe with metadata
        str mix_data --- relative path to audio mix / name of a corresponding column in dataframe
        List[str] ref_data --- relative paths to (ground truth) separate audio sources / 
                                                            names of corresponding columns in dataframe
        str audio_id --- name of the id column in dataframe
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
        super(SegmentedSeparationDataset, self).__init__(dataset_dir, csv_path, index)
        df = pd.read_csv(csv_path, usecols=csv_cols)
        if sample is not None:
            df = df.sample(sample)

        self.keys = df[audio_id].tolist()
        self.mode = mode
        mix_data = dict(zip(self.keys, df[mix_data].tolist()))
        ref_data = [dict(zip(self.keys, df[ref].tolist())) for ref in ref_data]
        self.preprocessing = preprocessing
        self.chunk_size = sample_rate * segment_length
        self.mix_data = []
        self.ref_data = []
        for i in tqdm(range(len(self.keys))):
            mix = self.load_audio(os.path.join(os.path.split(self.dataset_dir)[0], mix_data[self.keys[i]]))
            ref = [self.load_audio(os.path.join(os.path.split(self.dataset_dir)[0], ref[self.keys[i]])) for ref in ref_data]
            length = min([mix.shape[-1]] + [r.shape[-1] for r in ref])
            if self.mode == 'trim':
                cur_pos = 0
                while length >= self.chunk_size:
                    self.mix_data.append(mix[cur_pos:self.chunk_size + cur_pos])
                    self.ref_data.append([r[cur_pos:self.chunk_size + cur_pos] for r in ref])
                    length -= self.chunk_size
                    cur_pos += self.chunk_size
            else:
                if length >= self.chunk_size:
                    frame_offset = np.random.randint(0, length - self.chunk_size)
                    num_frames = self.chunk_size
                    mix = self.load_audio(
                        os.path.join(os.path.split(self.dataset_dir)[0], mix_data[self.keys[i]]),
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    )
                    ref = [self.load_audio(
                        os.path.join(os.path.split(self.dataset_dir)[0], ref[self.keys[i]]),
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    ) for ref in ref_data]
                    self.mix_data.append(mix)
                    self.ref_data.append([r for r in ref])
            

        if index is None:
            index = df.to_dict("records")

    def __len__(self):
        return len(self.mix_data)

    def __getitem__(self, key):
        mix = self.mix_data[key]
        ref = [r for r in self.ref_data[key]]
        if self.preprocessing is not None:
            for f in self.preprocessing.keys():
                mix = f(mix, **self.preprocessing[f])
                ref = [f(r, **self.preprocessing[f]) for r in ref]
        
        return mix, ref