import os, sys
import glob
import numpy as np
import torch
import pickle 
import random 
random.seed(1000)
class viDataLoader(object):
    def __init__(self, data_path, bsz, bptt, vocab, split_rate=0.8, device='cpu', ext_len=None, shuffle=True):
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.seq_len = bptt
        self.device = device
        self.vocab = vocab
        self.shuffle = shuffle        
        self.split_rate = split_rate
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f) # data format [num_sample, seq_len], default seq_len = 1024

    def _prepare_batches(self, batch_size=None, step=1):
        """
            prepare global batches from [samples, seq_len] to [seq_len]
        """
        global_batches = []
        batch_size = self.bsz if batch_size == None else batch_size
        self.bsz = batch_size
        # Process data loaded from pickle file
        # 
        
        for sample in self.data:
            try:
                last_idx = np.where(sample==self.vocab.pad_idx)[0][0]
            except:
                last_idx = sample.shape[0]
            if self.seq_len >= last_idx:
                continue
            else:
                for i in range(0, last_idx-self.seq_len-1, step):
                    x = sample[i:self.seq_len+i]
                    y = sample[i+1:self.seq_len+i+1]
                    global_batches.append([x,y])
        # shuffle data for training or not. Recommend shuffle == True
        if self.shuffle:
            random.shuffle(global_batches)       
        # convert to LongTensor
        self.global_batches = torch.LongTensor(global_batches)
        # delete variables for saving mem
        del global_batches
        
        self.n_step = self.global_batches.size(0) // batch_size
       

        # Trim off extra element that not cleanly fit with batch_size
        self.global_batches = self.global_batches.narrow(0, 0, self.n_step * batch_size)
        self.global_batches = self.global_batches.view(self.n_step, batch_size, 2, self.seq_len)
        # split train and test data batches base on split rate
        self.train_data = self.global_batches[:int(self.split_rate*self.n_step)]
        self.test_data = self.global_batches[int(self.split_rate*self.n_step):]
        self.train_step = self.train_data.size(0)
        self.eval_step = self.test_data.size(0)

    def get_batch(self, i, mode):
        if not hasattr(self, 'global_batches'):
            self._prepare_batches()
        if mode == 'train':
            mini_batch = self.train_data[i].permute(1,0,2) # permute from [bsz,'x,y', seq_len] -> ['x,y', bsz, seq_len]
        else: 
            mini_batch = self.test_data[i].permute(1,0,2)
        inp = mini_batch[0].t().contiguous().to(self.device)
        tgt = mini_batch[1].t().contiguous().to(self.device)
        # import pdb; pdb.set_trace()
        return inp, tgt, tgt.size(0)
    
    def _get_train_iter(self):
        for i in range(0, self.train_step):
            yield self.get_batch(i, mode = 'train')
    
    def _get_eval_iter(self):
        for i in range(0, self.eval_step):
            yield self.get_batch(i, mode = 'eval')

    def get_fixlen_iter(self, start=0, mode='train'):
        if mode == 'train':
            return self._get_train_iter()
        else:
            return self._get_eval_iter()

    # def __iter__(self):
    #     return self.get_fixlen_iter()

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)
        # self.data = 
        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()

