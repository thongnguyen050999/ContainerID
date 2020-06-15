### run script: python convert_hkt_to_music21 src_path dest_path
from pathlib import Path
from common.music_item import MusicItem
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

num_cores = multiprocessing.cpu_count()
src = Path(sys.argv[1])
dest = Path(sys.argv[2])

print(src, dest)

file_list = list(src.glob('**/*.json'))

total_files = len(file_list)
print('Number of files: {}'.format(total_files))

def convert_function(index):
    try:
        seq = MusicItem()
        seq.parse_from_hkt_file(str(file_list[index]))
        print(seq.write(dest / (file_list[index].stem + '.xml')))
    except:
        pass
processed_list = Parallel(n_jobs=8)(delayed(convert_function)(i) for i in tqdm(range(total_files)))