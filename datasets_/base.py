from torch.utils.data import Dataset
import torchaudio
from torch.utils.data import DataLoader

class BaseDataset(Dataset):
    '''
    Base Dataset for audio object
    Parameters
    ----------
    str dataset_dir --- path to the dataset root
    str csv_path --- path to the csv file with metadata
    List[dict] index --- custom index (if None, the df observations are used)
    '''
    def __init__(self, dataset_dir, csv_path, index=None):
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.index = index

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def load_audio(self, fname, return_sample_rate=False, frame_offset=0, num_frames=-1):
        audio_tensor, sample_rate = torchaudio.load(fname, frame_offset=frame_offset, num_frames=num_frames)
        if return_sample_rate:
            return audio_tensor.squeeze(), sample_rate
        return audio_tensor.squeeze()


def get_dataloader(dataset, data_config, batch_size=1, num_workers=0, is_train=True):
    '''
    Returns a DataLoader object based on the given config and dataset
    Parameters
    ----------
    dict data_config --- the config for SourceSeparationDataset object
    int batch_size --- the desired batch size (default is 1)
    int num_workers --- the desired num_workers value (default is 4)
    bool is_train --- whether the output dataset is used for training (default is True)
    '''
    return DataLoader(
        dataset=dataset(**data_config),
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers
    )