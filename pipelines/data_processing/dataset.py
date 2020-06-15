from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import os
import copy
import random
from sklearn.model_selection import train_test_split

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
class LyricsDataset(Dataset):
    """
    This class using to load the preprocess pickle
    """    
    def __init__(self,pkl_file,config):
        self.samples = None
        self.load_preprocess_dataset(pkl_file)

        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def shape(self):
        return [self.samples[0]['input'].shape,self.samples[0]['label'].shape]
    
    def load_preprocess_dataset(self, file_path):
        """
        Load pre-process dataset
        
        Args:
            
            file_path (str): File path to pickle file
        """
        with open(str(Path(file_path)), 'rb') as handle:
            self.samples = pickle.load(handle)

    def get_data(self,train_ratio = 0.8):
        inputs=[]
        results=[]

        random.shuffle(self.samples)

        for sample in self.samples:
            inputs.append(sample['input'])
            results.append(sample['label'])
    
        inputs=np.asarray(inputs)

        #Temporary padding: need to fix at Dataset Transform
        for i in range(len(results)):
            results[i]=np.pad(results[i],[(0,60-results[i].shape[0])],mode='constant')
        results=np.asarray(results)

        #Final data splitter
        train_data, val_data, train_label, val_label = train_test_split(inputs,results,test_size= 1 - train_ratio)

        return train_data,val_data,train_label,val_label


        