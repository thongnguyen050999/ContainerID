from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from common.music_item import XMLItem, MidiItem

class MusicDataset(Dataset):
    """
    This class using to load the preprocess pickle
    """    
    def __init__(self, pkl_file):
        self.samples = None
        self.load_preprocess_dataset(pkl_file)

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def shape(self):
        return self.samples.shape
    
    def load_preprocess_dataset(self, file_path):
        """
        Load pre-process dataset
        
        Args:
            
            file_path (str): File path to pickle file
        """
        with open(str(Path(file_path)), 'rb') as handle:
            self.samples = pickle.load(handle)

class MusicDatasetTransform():
    """
    Class to  manage and pre-process the music data
    
    Args:
    
        dataset_dir (str): Folder path to music score (xml) file
        transform (torchvision.transforms.Compose): Contains several transform pipeline stored into composed pipeline
        
    Return:
    
        pre-processed samples
    """
    def __init__(self, dataset_dir, file_handler=None, vocab=None, transform = None):
        self.dataset_dir = Path(dataset_dir)

        self.file_handler = file_handler
        self.vocab = vocab

        self.list_all_music_files()
        self.transform = transform

        self.samples = None

    def list_all_music_files(self):
        self.file_list = []

        for ext in ['xml', 'mxl', 'musicxml', 'mid', 'midi']:
            self.file_list += list(self.dataset_dir.glob('**/*.' + ext)) 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Read the score file

        sample = None
        # Data pre-processing

        file_path = self.file_list[idx]

        item = None
        if file_path.suffix in '.xml,.mxl,.musicxml':
            item = XMLItem()
        elif file_path.suffix == '.mid,.midi':
            item = MidiItem()

        item.from_file(file_path)

        handler = self.file_handler(item = item, vocab = self.vocab)

        if self.transform:
            sample = self.transform(handler)

        return sample

    def preprocess_dataset(self, num_workers=1):
        """
        Iterate through all the files and doing the transform
        """
        # self.file_list = list(self.dataset_dir.glob('**/*.xml'))
        self.list_all_music_files()

        # Get the sample shape
        single_sample_shape = None
        for idx in range(len(self)):
            try:
                single_sample_shape = self[idx].shape
                break
            except:
                pass
        
        samples = np.zeros((len(self),) + single_sample_shape, dtype=np.int16)
        valid_sample_indices = np.zeros(len(self), dtype=bool)
        
        inputs = tqdm(range(len(self)), desc="|-Pre-process dataset-|")
    
        def get_sample(idx):
            try:
                samples[idx] = self[idx]
                valid_sample_indices[idx] = True
            except Exception as e:
                pass
                # print('\n Error: {}. \n File name: {}'.format(str(e), self.file_list[idx]))
    
        for i in inputs:
            get_sample(i)
        
        # Reshape the array
        samples = samples[valid_sample_indices]
        new_shape = list(samples.shape)[1:]
        new_shape[0] = samples.shape[0] * new_shape[0] # Number of files * number of transposed
        
        self.samples = samples.reshape(new_shape)
        
        print('Samples generated. Samples size: {}, length: {}'.format(self.samples.shape[0], self.samples.shape[-1]))
    
    def save_preprocess_dataset(self, output_file, save_existing_samples=True, num_workers=1):
        """
        Pre-process the whole dataset and save the samples
    
        Args:

            output_file (str): File path to pickle file
            save_existing_samples (bool): If True, this will save the existing processed samples.
                                        If samples is empty, then pre-process the whole dataset
                                        If False, pre-process the dataset and save that samples
        """
        if (self.samples is None) or (not save_existing_samples):
            self.preprocess_dataset(num_workers)
            
        with open(str(Path(output_file)), 'wb') as handle:
            pickle.dump(self.samples, handle, protocol=pickle.HIGHEST_PROTOCOL)