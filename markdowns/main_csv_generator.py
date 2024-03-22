import os
import argparse
import sys
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def compute_csv_files(data_dir='./'):
    """
    Recursive function that join the morfeus properties inside separate .csv files into a single one.
    This should only be run when no other .csv files can be found in the folder and subdirectories
    """
    # Search each directory for .csv files
    os.chdir(data_dir)
    dirs = next(os.walk('./'))

    # Base case, process the found files or do nothing if there is none and there are no more directories
    res = pd.DataFrame()

    # Get the .csv files in this folder
    names = np.array(dirs[2])
    names = names[list(map(lambda x: x.endswith('.csv'), names))]
    names = list(map(lambda x: x[:-4], names))

    for n in names:
        logger.info(f'Now processing file {n}')
        df = pd.read_csv(f'{n}.csv')
        df.insert(0, 'file_name', n)  # Insert the file name for joining with log output later
        res = pd.concat([res, df], axis=0)

    # Recursive case, further directories
    if len(dirs[1]) > 0:
        prev_dir = os.getcwd()
        for d in dirs[1]:
            # Change the wd to the subdirectory
            res_rec = compute_csv_files(f'{prev_dir}/{d}')
            res = pd.concat([res, res_rec], axis=0)

        os.chdir(prev_dir)

    return res



if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description='Generate final .csv dataset from intermediary .csv files')
    parser.add_argument("--data_dir", type=str, help="Directory where to search for .csv files", default='./')

    args = parser.parse_args()

    logger.info('Started')

    res = compute_csv_files(data_dir=args.data_dir)
    res.reset_index(drop=True, inplace=True)
    res.to_csv('morfeus_values.csv', index=False)

    logger.info('Finished')
