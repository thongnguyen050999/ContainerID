from torchvision import transforms, utils
from common.object import HParams
import pipelines.data_processing.pipelines as pipelines
import collections

from common.music_item import ContainerItem
from common.vocab import ContainerVocabItem
from common.vocab_definition import CONTAINER

class Config(collections.namedtuple('Config', ['name', 'hparams', 'handler', 'vocab', 'transform_pipeline'])):

    def values(self):
        return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


container_config = Config(
    hparams=HParams(),
    handler= ContainerItem, # MusicItem handler
    vocab = ContainerVocabItem(CONTAINER.INDEX_TOKENS),
    transform_pipeline= transforms.Compose([
    ]
    )
)