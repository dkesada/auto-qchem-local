# Code partly adapted from https://github.com/doyle-lab-ucla/auto-qchem/blob/master/autoqchem/gaussian_log_extractor.py

import logging
import argparse
import sys
from autoqchem_local.api.api import AutoChem

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description='Generate dft .csv dataset from intermediary .log files')
    parser.add_argument("--data_dir", type=str, help="Directory where to search for .log files", default='./')

    args = parser.parse_args()

    logger.info('Started')

    controller = AutoChem()
    res = controller.process_log_files(data_dir=args.data_dir, output_path=args.data_dir)

    logger.info('Finished')
