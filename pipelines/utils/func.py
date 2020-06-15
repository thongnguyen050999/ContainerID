"""
Some functions and class used in making some processing for the data
"""
import collections
from math import pow

def text_and_melody_alignment(melody,text):
    """
    Align text and melody so that it can be sung naturally
    - Input:
        melody: a list of Note (defined in common/object.py)
        text: a list of syllable (Uppercase ones are strong-stressed syllables, lowercase one are weak-stressed ones)
    - Output:
        a list of Note, but with text attached
    """
    # threshold,strong_beat_note_index,weak_beat_note_index = otsu_melody(melody)
    return 0

def otsu_melody(self,melody):
    """
    Calculate optimal threshold to distinguish strong beat and weak beat
    Using Otsu's threshold for image processing
    - Input:
        a list of Note (common/object.py)
    - Output:
        threshold pitch
        strong beat index
        weak beat index
    - reference link: https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    #1. Filtering notes so that it only contains high notes for singing
    group_sequence = collections.defaultdict(list)
    for i,note in enumerate(melody):
        group_sequence[note.start].append((note,i))
    highest_note_index_by_timestep = [max(group_sequence[x],lambda x : x[0].pitch)[1] for x in group_sequence.keys()]
    #2. Otsu's method to find threshold, dividing melody to strong-beat ones and weak-beat ones
    #find upper and lower bound
    upper_bound = max([melody[x].pitch for x in highest_note_index_by_timestep])
    lower_bound = min([melody[x].pitch for x in highest_note_index_by_timestep])
    #create a histogram dictionary
    histogram = dict()
    for index in highest_note_index_by_timestep:
        if index not in histogram:
            histogram[index] = 0
        histogram[index] += 1
    n = len(highest_note_index_by_timestep)
    optimal_threshold = -1
    max_variance = 0
    for threshold in range(lower_bound,upper_bound + 1):
        x_background = float(sum([histogram[x] for x in highest_note_index_by_timestep if x <= threshold]))
        x_foreground = float(sum([histogram[x] for x in highest_note_index_by_timestep if x >= threshold]))
        #calculate w
        w_background = x_background / (n * n)
        w_foreground = x_foreground / (n * n)
        #calculate muy
        muy_background = float(sum([histogram[x] * x for x in highest_note_index_by_timestep if x <= threshold])) / x_background
        muy_foreground = float(sum([histogram[x] * x for x in highest_note_index_by_timestep if x >= threshold])) / x_foreground
        #calculate variance
        variance_background = float(sum([histogram[x] * pow(x - muy_background,2) for x in highest_note_index_by_timestep if x <= threshold])) / x_background
        variance_foreground = float(sum([histogram[x] * pow(x - muy_foreground,2) for x in highest_note_index_by_timestep if x <= threshold])) / x_background
        #total variance
        variance = w_background * variance_background + w_foreground * variance_foreground
        if max_variance < variance:
            variance = max_variance
            optimal_threshold = threshold

    strong_beat_note_index = [i for i in highest_note_index_by_timestep if melody[i].pitch >= threshold]
    weak_beat_note_index = [i for i in highest_note_index_by_timestep if melody[i].pitch < threshold] #prevent overlapping

    #strong_beat + weak_beat >= len(text)
    return optimal_threshold,strong_beat_note_index,weak_beat_note_index