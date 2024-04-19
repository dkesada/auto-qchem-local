import argparse
import sys
import logging
import pandas as pd


logger = logging.getLogger(__name__)


def join_csv_files(log_dir='./log_values.csv', morfeus_dir='./morfeus_values.csv'):
    """
    Function that joins the log extracted properties and the morfeus properties inside separate .csv files into a single
    one. They will be joined by the file_name column, which contains the original names of the .log and .smi files.
    Both of these should have the same name for this join to work.
    """
    # Load both .csv files
    df_log = pd.read_csv(log_dir)
    df_morfeus = pd.read_csv(morfeus_dir)

    # Join them by the 'file_name' column
    res = pd.merge(df_log, df_morfeus, on='file_name', how='outer')

    return res


if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description='Generate final joined .csv dataset from intermediary morfeus and log .csv files')
    parser.add_argument("--log_dir", type=str, help="Path to the log_values.csv file", default='./log_values.csv')
    parser.add_argument("--morfeus_dir", type=str, help="Path to the morfeus_values.csv file", default='./morfeus_values.csv')

    args = parser.parse_args()

    logger.info('Started')

    res = join_csv_files(log_dir=args.log_dir, morfeus_dir=args.morfeus_dir)
    res.reset_index(drop=True, inplace=True)
    res.to_csv('join_data.csv', index=False)

    logger.info('Finished')
