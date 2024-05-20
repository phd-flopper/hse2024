from datasets_.base import BaseDataset
import pandas as pd


class ClassifierDataset(BaseDataset):
    def __init__(self, dataset_dir, csv_path, target, _object, csv_cols, encoding='utf-8',
                 sep=',', index=None, preprocessing=None):
        '''
        Parameters
        ----------
        str dataset_dir --- path to dataset root
        type target --- column id of the target
        type object --- column if of the object
        str csv_path --- path to dataframe with metadata
        List[str] csv_cols --- list of dataframe columns to consider (if None, all columns are considered)
        dict[callable : dict] preprocessing --- dict with preprocessing functions as keys and 
                            their respective arguments as values (if None, labels are not encoded)
        '''

        df = pd.read_csv(csv_path, usecols=csv_cols, encoding=encoding, sep=sep)
        self.preprocessing = preprocessing
        self.objects = df[_object].tolist()
        self.targets = df[target].tolist()

        if index is None:
            index = df.to_dict("records")

        super(ClassifierDataset, self).__init__(dataset_dir, csv_path, index)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, key):
        return self.objects[key], self.targets[key]