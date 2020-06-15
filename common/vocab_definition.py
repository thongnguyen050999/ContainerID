import itertools

def add_dummy_tokens(itos):
    """
    The size of total vocab should be multiple of 8
    """
    if len(itos) % 8 != 0:
        itos = itos + [f'dummy{i}' for i in range(len(itos) % 8)]
    return itos


def vocab_list_generator(vocab):
    itos = []
    for v in vocab:
        itos.append(v.to_list())
    itos = list(itertools.chain.from_iterable(itos))
    return itos

# Definition of one event as vocab


class VocabEvent():
    """Class to handle vocab event"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return "Vocab Event: name = '{}', prefix = '{}', size = '{}' \n".format(self.name, self.prefix, self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Get item forward and backward

        If idx is number, convert number into the vocab text. For e.g: idx = 0 => {self.prefix}{idx}
        If idx is string, convert string vocab into index. For e.g: {self.prefix}{idx} => idx
        """
        if isinstance(idx, int):
            if idx >= len(self):
                raise ValueError(
                    "{} - Index out of range. Object's length: {}".format(self.name, len(self)))

            indices = [i for i in range(len(self))]
            return '{}{}'.format(self.prefix, indices[idx] if len(self) > 1 else '')
        elif isinstance(idx, str):
            index = int(idx.replace(self.prefix, ''))
            return index

    def to_list(self):
        return [self[idx] for idx in range(len(self))]

class CONTAINER:
    # Basic vocab
    NUM = VocabEvent(name='Num', prefix='n', size=10)
    CHAR = VocabEvent(name='Char', prefix='c', size=26)
    PAD = VocabEvent(name='Pad',prefix='pad',size=1)

    # Total vocab list
    VOCAB = [NUM,CHAR,PAD]
    INDEX_TOKENS = vocab_list_generator(VOCAB)