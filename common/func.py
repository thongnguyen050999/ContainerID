"""
Some functions and class used in making some processing for the data
"""
import collections
from math import pow
import re
from common.constants import COMPOSER
from common.object import Note
import unicodedata
import nltk

def otsu_melody(bar_seq):

    """
    Calculate optimal threshold to distinguish strong beat and weak beat
    Using Otsu's threshold for image processing
    - Input:
        a list of (Note,bar_index,pos_index)
    - Output:
        threshold pitch
        strong beat index
        weak beat index
    - reference link: https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    #1. Filtering notes so that it only contains high notes for singing
    group_sequence = collections.defaultdict(list)

    for bar_idx,bar in enumerate(bar_seq):
        for obj_idx,obj in enumerate(bar):
            if isinstance(obj,Note):
                group_sequence[(bar_idx,obj.start)].append((obj,(bar_idx,obj_idx)))
    highest_note_tuple_index_by_timestep = [max(group_sequence[x],key=lambda y : y[0].pitch)[1] for x in sorted(group_sequence.keys())]
    #2. Otsu's method to find threshold, dividing bar_seq to strong-beat ones and weak-beat ones
    #find upper and lower bound
    upper_bound = max([bar_seq[x[0]][x[1]].pitch for x in highest_note_tuple_index_by_timestep])
    lower_bound = min([bar_seq[x[0]][x[1]].pitch for x in highest_note_tuple_index_by_timestep])
    #create a histogram dictionary
    histogram = dict()
    for ti in highest_note_tuple_index_by_timestep:
        if bar_seq[ti[0]][ti[1]].pitch not in histogram:
            histogram[bar_seq[ti[0]][ti[1]].pitch] = 0
        histogram[bar_seq[ti[0]][ti[1]].pitch] += 1
    n = len(highest_note_tuple_index_by_timestep)
    optimal_threshold = -1
    min_variance = float('inf')
    for threshold in range(lower_bound,upper_bound + 1):
        x_background = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch <= threshold]))
        x_foreground = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch >= threshold]))
        #calculate w
        w_background = x_background / n
        w_foreground = x_foreground / n
        #calculate muy
        muy_background = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] * bar_seq[x[0]][x[1]].pitch for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch <= threshold])) / x_background
        muy_foreground = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] * bar_seq[x[0]][x[1]].pitch for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch >= threshold])) / x_foreground
        #calculate variance
        variance_background = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] * pow(bar_seq[x[0]][x[1]].pitch - muy_background,2) for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch <= threshold])) / x_background
        variance_foreground = float(sum([histogram[bar_seq[x[0]][x[1]].pitch] * pow(bar_seq[x[0]][x[1]].pitch - muy_foreground,2) for \
            x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch >= threshold])) / x_foreground
        #total variance
        variance = w_background * variance_background + w_foreground * variance_foreground
        if min_variance > variance:
            min_variance = variance
            optimal_threshold = threshold
    strong_beat_note_index = [x for x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch >= optimal_threshold]
    weak_beat_note_index = [x for x in highest_note_tuple_index_by_timestep if bar_seq[x[0]][x[1]].pitch < optimal_threshold] #prevent overlapping
    return optimal_threshold,strong_beat_note_index,weak_beat_note_index