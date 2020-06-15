import copy
import torch
import numpy as np
from abc import ABCMeta, abstractmethod, ABC
import abc

class SequenceError(Exception):
    def __init__(self, msg):
        self.message = msg

class Pipeline(ABC):
    def __init__(self):
        pass
    @abc.abstractmethod
    def __call__(self,sample):
        #TODO: Implement this in inherited class
        pass
        
class IsTimeSignature44(Pipeline):
    """
    Only take the score with 4/4 time signature
    """
    def __init__(self):
        pass
    
    def __call__(self, sample):
        time_sig = sample.item.time_signature
        
        if (time_sig.numerator == 4) and (time_sig.denominator == 4):
            return sample
        else:
            raise ValueError("Sequence is in {} / {} time signature".format(time_sig.numerator, time_sig.denominator))
        
class IsValidMelody(Pipeline):
    """
    Check whether melody part is empty or contains only a few notes
    
    Args:
        
        sample (MusicItem):
        min_notes_per_bar (int): Minimum mumber of notes per bar
    """
    
    def __init__(self, min_notes_per_bar):
        self.min_notes_per_bar = min_notes_per_bar
        
    def __call__(self, sample):
        highest_time = sample.item.score.flat.getElementsByClass(['Note','Chord']).highestTime
        num_bars = highest_time*CONVERSION.quarterLength_to_bar_ratio()

        num_notes = len(sample.item.score.flat.getElementsByClass('Note'))
        average_notes_per_bar = num_notes / num_bars

        if (average_notes_per_bar > self.min_notes_per_bar):
            return sample
        else:
            raise SequenceError('Sequence has too few notes. Has average {} notes per bar'.format(average_notes_per_bar))

class SameLengthForAllTrack(Pipeline):
    """
    There're some time the chord part is longer than the melody part
    Trim the longer one to make it equivalent
    """
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        if sample is None: return sample
        
        item = copy.deepcopy(sample)

        time_signature = item.score.parts[DEF.MELODY_PART_INDEX].flat.getElementsByClass('TimeSignature')[0]
        duration_per_bar = CONVERSION.bar_to_quarterLength_ratio(time_signature)
        melody_highestTime = item.score.parts[DEF.MELODY_PART_INDEX].getElementsByClass('Measure')[-1].offset + duration_per_bar
        chord_highestTime = item.score.parts[DEF.CHORD_PART_INDEX].getElementsByClass('Measure')[-1].offset + duration_per_bar

        for part in item.score.parts:
            for element in part.getElementsByClass('Measure'):
                if element.offset > min(melody_highestTime,chord_highestTime):
                    part.remove(element)

        return item
    
class DataCompression(Pipeline):
    """
    Compress and simplify the data by turning into no-sharp key signature
    
    """
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        if sample is None: return sample

        compressed_sample = copy.deepcopy(sample)
        compressed_sample.item.transpose_to_nosharp_key(in_place=True)
        
        return compressed_sample
    
class DataAugmentation(Pipeline):
    """
    Transpose with one octave interval
    
    Args:
        
        transpose_range (list): List of number octave want to transpose
        
    Returns:
    
        list: List of transposed MusicItem
    """
    def __init__(self, transpose_range, is_octave_range = True):

        self.transpose_range = transpose_range
        self.is_octave_range = is_octave_range

    def __call__(self, sample):
        if sample is None: return sample

        samples = []
        for times in self.transpose_range:
            transposed_sample = copy.deepcopy(sample)
            
            if self.is_octave_range:
                transposed_sample.item.transpose_octave(times, in_place=True)
            else:
                transposed_sample.item.transpose(times, in_place=True)

            samples.append(transposed_sample)

        return samples
    
class Tokenizing(Pipeline):
    """
    Convert MusicItem object into tokens
    
    Args:
        list or single object of MusicItem
    
    Returns:
    
        list: List of ( tokens for each MusicItem)
    """
    def __init__(self, padding_len=1024):
        self.padding_len = padding_len
    
    def __call__(self, sample):
        if sample is None: return []

        tokens = []
        
        if isinstance(sample, list):
            for sampl in sample:
                tokens.append(self.tokenize(sampl, self.padding_len))
        else:
            tokens.append(self.tokenize(sample, self.padding_len))

        return np.array(tokens)
        
    def tokenize(self, item, padding_len):
        return item.to_tokens(padding_len=padding_len)
    
class PositionalTokenizing(Pipeline):
    """
    Convert MusicItem object into tokens
    
    Args:
        list or single object of MusicItem
    
    Returns:
    
    """
    def __init__(self, padding_len=1024):
        self.padding_len = padding_len
    
    def __call__(self, sample):
        if sample is None: return []

        tokens = []
        if isinstance(sample, list):
            for sampl in sample:
                tokens.append(self.tokenize(sampl, self.padding_len))
        else:
            tokens.append(self.tokenize(sample, self.padding_len))

        return np.array(tokens)
        
    @classmethod
    def tokenize(cls, item, padding_len):
        pos_enc_tks = item.to_positional_melody_chord_tokens(padding_len)
        
        return pos_enc_tks
    
class MultitaskTokenizing(PositionalTokenizing):
    def __init__(self, padding_len=1024):
        super().__init__(padding_len + 1)
        
class MultitaskDataShift(Pipeline):
    """ Sample output dimention:
            - number of transposed item
            - number of tokens array (melody, chord)
            - tokens type. With x_dict, store tokens (0), pos_enc (1), shifted tokens (2)
            - tokens length
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        seq_len = sample.shape[-1] -1
        shape = list(sample.shape)
        
        shape[2] += 1; shape[-1] -= 1        
        shifted_sample = np.zeros(shape)
        
        shifted_sample[:, :, :2, :] = sample[:, :, :, :-1]
        
        for idx in range(shifted_sample.shape[0]):
            shifted_sample[idx, 0, 2, :] = sample[idx, 0, 0, 1:]
            shifted_sample[idx, 1, 2, :] = sample[idx, 1, 0, 1:]
        
        return shifted_sample

class LyricsSegment(Pipeline):
    def __init__(self,w_size,max_ssm_size):
        self.w_size = w_size
        self.max_ssm_size = max_ssm_size

    def __call__(self,sample):
        """
        Input: a LyricsItem
        Output: a list of segment with label
        """
        #1. From the lyrics, find out indices of sentences that mark the end of the segment
        current_len = 0
        segment_indicies = []
        for segment in sample.lyrics:
            segment_indicies.append(current_len + len(segment) - 1)
            current_len += len(segment)

        samples = []
        ssm_input = copy.deepcopy(sample.input)
        max_size = ssm_input.shape[0]
        if ssm_input.shape[0] > self.max_ssm_size:
            ssm_input = ssm_input[:self.max_ssm_size,:self.max_ssm_size]
            max_size = self.max_ssm_size
        elif ssm_input.shape[0] < self.max_ssm_size:
            pad_value = self.max_ssm_size - ssm_input.shape[0]
            ssm_input = np.pad(ssm_input,((0,pad_value),(0,pad_value)),mode='constant')

        for index in range(max_size):
            sample_dict = {
                "input" : [],
                "label" : 0
            }
            line_range = range(index - self.w_size + 1,index + self.w_size + 1)
            for i in line_range:
                if i < 0:
                    result_line = np.zeros(self.max_ssm_size)
                else:
                    result_line = copy.deepcopy(ssm_input[index])
                sample_dict["input"].append(result_line)
            
            if index in segment_indicies:
                sample_dict["label"] = 1
            
            sample_dict["input"] = np.asarray(sample_dict["input"])

            #append to samples
            samples.append(sample_dict)

        return samples


