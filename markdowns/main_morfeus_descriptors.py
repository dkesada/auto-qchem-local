import os
import argparse
import logging
from datetime import datetime

from morfeus_ml.morfeus_descriptors import *
from morfeus import read_xyz


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def extract_properties_compound(file_name, output_path, n_confs, solvent):

    # Read the .smi file
    smi_file = open(f'{file_name}.smi', 'r')
    smiles = smi_file.readlines()  # This should only contain a single smile
    smi_file.close()
    smiles = smiles[0]
    smiles = smiles.replace('\n', '')

    # Read the .xyz file
    elements, coords = read_xyz(f'{file_name}.xyz')
    conf = Conformer(elements, coords)

    # Create the conformer ensemble
    ce = compute_with_xyz(smiles, conf, n_confs=n_confs, solvent=solvent)

    # Calculate the descriptors of the ensemble
    descs = get_descriptors(ce)

    os.makedirs(os.path.dirname(f"{output_path}/{file_name}_descriptors.csv"), exist_ok=True)
    with open(f"{output_path}/{file_name}_descriptors.csv", "wb") as f:
        descs.to_frame(smiles).T.to_csv(f, index_label="smiles")


if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    date = datetime.now()
    file_log_handler = logging.FileHandler(filename=date.strftime('morfeus_%d_%m_%Y_%H_%M.log'), mode='w')
    logger.addHandler(file_log_handler)

    logger.info('Started')
    logger.info(f'Current time: {datetime.now()}')

    parser = argparse.ArgumentParser(description='Compute Conformational Ensemble and its Features')
    parser.add_argument("file_name", type=str, help="Name of the files .xyz and .smi files")
    parser.add_argument("output_path", type=str, help="Storage output path")
    parser.add_argument("--n_confs", type=int, help="Optional number of conformers to initially generate with RDKit. "
                                                    "If not specified a default is used based on the number of"
                                                    " rotatable bonds (50 if n_rot <=7, 200 if n_rot <=12, "
                                                    "300 for larger n_rot)",
                        default=None)
    parser.add_argument("--solvent", type=str, help="XTB supported solvents and their names can be found at "
                                                    "https://xtb-docs.readthedocs.io/en/latest/gbsa.html",
                        default=None)

    args = parser.parse_args()

    extract_properties_compound(args.file_name, args.output_path, args.n_confs, args.solvent)

    logger.info('Finished')
    logger.info(f'Current time: {datetime.now()}')
