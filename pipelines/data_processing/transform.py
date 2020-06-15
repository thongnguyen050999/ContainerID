from abc import ABCMeta, abstractmethod, ABC
import abc
import random
import os
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import lyricwikia

class DatasetTransform(ABC):

    def __init__(self, dataset_dir, file_handler=None, vocab=None, transform = None):
        self.dataset_dir = dataset_dir
        self.file_handler=file_handler
        self.vocab = vocab
        self.transform = transform
        self.samples = None

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self):
        pass
    
    @abc.abstractmethod
    def preprocess_dataset(self,num_workers=1):
        pass

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


class MusicDatasetTransform(DatasetTransform):
    """
    Class to  manage and pre-process the music data
    
    Args:
    
        dataset_dir (str): Folder path to music score (xml) file
        transform (torchvision.transforms.Compose): Contains several transform pipeline stored into composed pipeline
        
    Return:
    
        pre-processed samples
    """
    def __init__(self, dataset_dir, file_handler=None, vocab=None, transform = None):
        super(MusicDatasetTransform,self).__init__(
            dataset_dir,
            file_handler, 
            vocab, 
            transform
        )

        self.file_list = list(self.dataset_dir.glob('**/*.xml'))
        
        self.samples = None
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        # Read the score file
      
        sample = None
        # Data pre-processing
        item = self.file_handler(vocab = self.vocab)
        item.from_file(self.file_list[idx])
        
        if self.transform:
            sample = self.transform(item)

        return sample
    
    def preprocess_dataset(self, num_workers=1):
        """
        Iterate through all the files and doing the transform
        """
        self.file_list = list(self.dataset_dir.glob('**/*.xml'))

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

class LyricsDatasetTransform(DatasetTransform):
    """
    Class to preprocess lyrics data into w_size segments

    Args:
        dataset_dir (str): folder to dataset contianing lyrics
        transform (torchvision.transforms.Compose)

    return:

        Pre-process dataset
    """
    def __init__(self,dataset_dir,file_handler=None,transform=None,max_parts=2,max_songs=10,shuffle=False,min_ssm_size=30, max_ssm_size=60):
        super(LyricsDatasetTransform,self).__init__(
            dataset_dir,
            file_handler, 
            None, 
            transform
        )
        self.max_parts = max_parts

        self.obj_list = dict()
        #read ssm data
        for i in range(self.max_parts):
            try:
                ssm = pd.read_pickle(os.path.join(
                    self.dataset_dir,
                    "ssm_wasabi",
                    "ssm_wasabi_{}.pickle".format(i)))
                self.obj_list.update(ssm)
            except:
                print("Corrupted ssm_wasabi_{}.pickle. Skip.".format(i))

        if len(self.obj_list) == 0:
            raise RuntimeError("All ssm file corrupted.")

        #add keys to dict and convert to list
        for key in self.obj_list:
            self.obj_list[key]["id"] = key
        self.obj_list = list(self.obj_list.values())

        #read wasabi songs
        self.wasabi_info = pd.read_csv(os.path.join(
            self.dataset_dir,
            "wasabi_songs.csv"
        ),sep="\t")

        self._max_parts = max_parts #default

        self._max_songs = max_songs
        if self._max_songs > len(self.obj_list):
            print("Max song exceeds number of obj. resetting max songs")
            self._max_songs = len(self.obj_list)

        self.shuffle = shuffle

        self.min_ssm_size = min_ssm_size

        self.max_ssm_size = max_ssm_size

    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self,idx):
        """
        Return a list of segment
        """
        if self.shuffle:
            #FIXME: Do we need to find better ways?
            self.obj_list = random.shuffle(self.obj_list)

        #Check if id of idx exist
        obj = self.obj_list[idx]
        id = obj['id'] #id
        ssm = obj['line'] #numpy matrix
        if ssm.shape[0] != ssm.shape[1]:
            return None
        if ssm.shape[0] < self.min_ssm_size:
            return None
        segment_ssm = obj['segment'] #numpy segment matrix
        song_row = self.wasabi_info[self.wasabi_info['_id'] == id]
        if len(song_row) > 1:
            print("Warning: More than one row information. Choose randomly")
            info_idx = random.choice(range(len(song_row)))
        elif len(song_row) == 1:
            info_idx = 0
        else:
            print("No information found. Return None")
            return None
        artist = song_row['artist'].values[info_idx]
        song = song_row['title'].values[info_idx]

        try:
            item = self.file_handler(
                id,
                artist,
                song,
                ssm,
                segment_ssm)
        except:
            print("Something wrong with this id. Returning None.")
            return None

        if self.transform:
            samples = self.transform(item)

            input = np.asarray([x['input'] for x in samples])
            if input.shape[0] <= self.max_ssm_size:
                input = np.pad(input,\
                    [(0,self.max_ssm_size - input.shape[0]),(0,0),(0,0)],\
                        mode='constant')
            label = np.asarray([x['label'] for x in samples])
            if label.shape[0] <= self.max_ssm_size:
                label = np.pad(label,(0,self.max_ssm_size - label.shape[0]),mode='constant')
            return {
                "input" : input,
                "label" : label
            }
    
        return None

    def preprocess_dataset(self,num_workers=1):
        """
        Iterating through all the files and doing the transform
        """

        # Get the sample shape
        self.samples = []
        for i in range(self._max_songs):
            print(i)
            samples = self[i]
            if samples is not None: self.samples.append(samples)
               

               
            
