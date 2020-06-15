import copy
from pathlib import Path
import os, sys
sys.path[0] = os.getcwd()

from pipelines.data_processing.configs import test_config,lyrics_segmentation_config
from pipelines.data_processing.transform import LyricsDatasetTransform, MusicDatasetTransform
from pipelines.data_processing.configs import remi_all_key_config, remi_all_key_and_transpose_config, remi_no_sharp_config, remi_no_sharp_and_transpose_config

dataset_dir = Path('models/remi/dataset_v2')

# dataset_transform.preprocess_dataset(num_workers=10)
# output_file = Path("models/remi/datasets.pkl")
# dataset_transform.save_preprocess_dataset(output_file)

lyrics_seg_dir = Path('models/lyrics_segmentation/datasets')
lyrics_dataset_transform = LyricsDatasetTransform(
    lyrics_seg_dir,
    lyrics_segmentation_config.handler,
    lyrics_segmentation_config.transform_pipeline)
lyrics_dataset_transform.save_preprocess_dataset(2)



for config in [remi_no_sharp_config]:
    print('Running config: ' + config.name)
    dataset_transform = MusicDatasetTransform(dataset_dir, config.handler, config.vocab, config.transform_pipeline)
    dataset_transform.preprocess_dataset()
    output_file = Path('models/remi/datasets_' + config.name + '.pkl')
    dataset_transform.save_preprocess_dataset(output_file)
