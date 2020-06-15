import pickle
from common.vocab_definition import CONTAINER
# Vocab - token to index mapping


class VocabItem():
    "Contain the correspondence between numbers and tokens and numericalize."

    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v: k for k, v in enumerate(self.itos)}

    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        items = [self.itos[i] for i in nums]
        return sep.join(items) if sep is not None else items

    def __getstate__(self):
        return {'itos': self.itos}

    def __setstate__(self, state: dict):
        self.itos = state['itos']
        self.stoi = {v: k for k, v in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)  

class ContainerVocabItem(VocabItem):
    @property
    def pad_idx(self): return self.stoi[CONTAINER.PAD.prefix]