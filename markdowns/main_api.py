import sys
import logging
import argparse
import datetime

from autoqchem_local.api.api import AutoChem

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Conformational Ensemble and its Features')
    parser.add_argument("--data_dir", type=str, help="Directory where to search for .smi files", default='./')
    parser.add_argument("--smiles_file", type=str, help="Path to the .smi file to generate all .gjf files", default='./smiles.smi')
    parser.add_argument("--output_path", type=str, help="Storage output path", default='jobs')
    parser.add_argument("--n_confs", type=int, help="Optional number of conformers to initially generate with RDKit. "
                                                    "If not specified a default is used based on the number of"
                                                    " rotatable bonds (50 if n_rot <=7, 200 if n_rot <=12, "
                                                    "300 for larger n_rot)",
                        default=5)
    parser.add_argument("--solvent", type=str, help="XTB supported solvents and their names can be found at "
                                                    "https://xtb-docs.readthedocs.io/en/latest/gbsa.html",
                        default=None)

    args = parser.parse_args()

    logger.info('Started')

    controller = AutoChem(log_to_file=True)
    # controller.process_log_files(data_dir=args.data_dir, output_path=args.data_dir)
    controller.generate_gjf_files(args.smiles_file)

    logger.info('Finished')
