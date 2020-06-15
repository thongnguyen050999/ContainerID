import copy
import json

class Dict(dict):
    """
        Dictionary processing method
    """

    def read_from_file(self, fp):
        with open(fp, 'r') as f:
            self.update(json.load(f))

    def write_to_file(self, fp):
        with open(fp, 'w') as f:
            json.dump(self, f)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class Configs(Dict):
    """Class to handle configs"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def merge(self, other_dict, in_place=False):
        """Merge two config"""
        assert isinstance(other_dict, dict)
        og_dict = copy.deepcopy(self)
        og_dict.update(other_dict)

        if not in_place:
            return og_dict
        else:
            self.__init__(**og_dict)

    def __add__(self, other_dict):
        assert isinstance(other_dict, dict)
        return self.merge(other_dict, in_place=False)


class HParams(Configs):
    """Class to handle parameters"""
    pass