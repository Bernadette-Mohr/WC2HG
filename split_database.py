import argparse
import sys
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


def chunk_list(lst, n, start_idx=0):
    """Yield successive n-sized chunks from a list."""
    for i in range(start_idx, len(lst), n):
        yield lst[i:i + n]


def wrapper(gen, fname, chunk):
    for idx in tqdm(chunk, desc=f'Reading steps'):
        try:
            yield gen[idx]
        except StopIteration:
            break
        except Exception as e:
            print(f'Unable to load step {idx} from {fname}.db: {e.__class__}: {e}')


def split_database(directory, db_name, l_chunks, start_idx):
    if Path(db_name).suffix == '.db':
        db_name = Path(db_name).stem
    storage = Storage(filename=f'{directory}/{db_name}.db', mode='r')
    len_db = len(storage.steps) + 1
    print(f'Length of database: {len_db}')
    db_chunks = chunk_list(range(len_db), l_chunks, start_idx)
    # print(*db_chunks)
    # sys.exit()

    for idx, chunk in enumerate(db_chunks):
        """
         Needs testing: Automatic adjustment of file index in case of restart.
        """
        if start_idx > 0:
            idx += int(start_idx / l_chunks)
        print(idx)
        # new_storage = Storage(filename=f'{directory}/{db_name}_{idx}.db', mode='w')
        # steps = wrapper(storage.steps, db_name, chunk)
        # for step in steps:
        #     new_storage.save(step)
        # new_storage.close()
    storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split a big TPS output database file into chunks of 200.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output.')
    parser.add_argument('-fn', '--file_name', type=str, required=True,
                        help='Name of the database file to be split into chunks.')
    parser.add_argument('-l', '--length_of_chunks', type=int, required=True,
                        help='Name of the new OPS database file with all accepted trials.')
    parser.add_argument('-s', '--start', type=int, required=False, default=0,
                        help='Start index for splitting in case of restart.')

    args = parser.parse_args()
    source_dir = args.directory
    file_name = args.file_name
    length_of_chunks = args.length_of_chunks
    start = args.start
    split_database(source_dir, file_name, length_of_chunks, start)
