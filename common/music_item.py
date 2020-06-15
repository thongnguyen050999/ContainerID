import os
import shutil

import copy
import numpy as np
# import common.vocab as vocab_lib
from abc import ABCMeta, abstractmethod, ABC
import re

class ContainerItem(object):

	def __init__(self, item=None, vocab=None):
		self.item = item
		self.vocab = vocab