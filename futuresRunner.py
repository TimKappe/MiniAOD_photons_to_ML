import concurrent.futures
import argparse
import os

from preprocess_futures import process_file

from typing import Tuple


def process_args() -> Tuple[str, int]:
    parser = argparse.ArgumentParser(description='manage multicore threading for preprocess_futures',
                                     prog='futureRunner.py')
    parser.add_argument('file', help='file containing the names of the dataset files')
    parser.add_argument('workers', help='set number of cores, (set to 24 on lxblade)')
    parser.add_argument('--exclude', help='exclude files in this directory')
    parser.add_argument('--skip', help='skip the first n files (after ignore is applied)')
    parser.add_argument('--shuffle', action='store_true', help='shuffle input file list')

    args = parser.parse_args()
    print(args)
    file_: str = args.file
    workers_: int = int(args.workers)
    ignore_dir: str = args.exclude

    files_ = read_filenames(file_)

    if ignore_dir is not None:
        ignore_list = os.listdir(ignore_dir)
        for name in ignore_list:
            name = name.split('.')[0]
            for i, filename in enumerate(files_):
                if filename.split('.')[0].split('/')[-1] == name: 
                    files_.pop(i)
                    break

    skip = args.skip
    if skip is not None:
        for i in range(int(skip)):
            files_.pop(0)
    if args.shuffle:
        import random
        random.shuffle(files_)
        print('list shuffled')
    print(f'INFO: starting to process {len(files_)} files')
    return files_, workers_

def read_filenames(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def main(filenames, workers=4):
    total_files = len(filenames)
    completed = 0  # Initialize a counter for completed tasks

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, filename): filename for filename in filenames}
        
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            completed += 1  # Increment the counter when a task is completed
            try:
                future.result()
                print(f'INFO: {filename} processed successfully. [{completed}/{total_files} completed]')
            except Exception as exc:
                print(f'ERROR: {filename} generated an exception: {exc}. [{completed}/{total_files} completed]')


if __name__ == '__main__':
    filenames, workers = process_args()
    main(filenames, workers=workers)
