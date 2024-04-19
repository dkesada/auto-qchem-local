# Code partly adapted from https://github.com/doyle-lab-ucla/auto-qchem/blob/master/autoqchem/sge_manager.py

import logging
from datetime import datetime
import sys
from tqdm import tqdm
import func_timeout

from autoqchem_local.autoqchem.molecule import molecule
from autoqchem_local.autoqchem.gaussian_input_generator import *


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_gjf_for_molecule(smiles, workdir='./output_gjf', workflow_type="custom", theory="B3LYP", solvent="None",
                            light_basis_set="6-31+G(d,p)", heavy_basis_set="SDD", generic_basis_set="genecp",
                            max_light_atomic_number=25) -> None:
    """
    Generate Gaussian input files for a molecule

    :param smiles: a SMILES code from a molecule
    :type smiles: str
    :param workdir: output directory for the .gjf files
    :type workdir: str
    :param workflow_type: Gaussian workflow type, allowed types are: 'equilibrium' or 'transition_state'
    :type workflow_type: str
    :param theory: Gaussian supported Functional (e.g., APFD, B3LYP)
    :type theory: str
    :param solvent: Gaussian supported Solvent (e.g., TETRAHYDROFURAN)
    :type solvent: str
    :param light_basis_set: Gaussian supported basis set for elements up to `max_light_atomic_number` (e.g., 6-31G*, 6-31+G(d,p))
    :type light_basis_set: str
    :param heavy_basis_set: Gaussian supported basis set for elements heavier than `max_light_atomic_number` (e.g., LANL2DZ)
    :type heavy_basis_set: str
    :param generic_basis_set: Gaussian supported basis set for generic elements (e.g., gencep)
    :type generic_basis_set: str
    :param max_light_atomic_number: maximum atomic number for light elements
    :type max_light_atomic_number: int
    """

    # create gaussian files
    try:
        m = func_timeout.func_timeout(40, lambda: molecule(smiles, num_conf=1))
        simplified_name = ''.join(e for e in smiles if e.isalnum())
        molecule_workdir = os.path.join(f"{workdir}", simplified_name)
        gig = gaussian_input_generator(m, workflow_type, workdir, theory, solvent, light_basis_set,
                                       heavy_basis_set, generic_basis_set, max_light_atomic_number)
        gig.create_gaussian_files()
        with open(f'output_gjf/{m.inchikey}_conf_0.gjf.smi', "w") as f:
            f.write(smiles)

    except func_timeout.FunctionTimedOut:
        logger.error(f"Timed out! Possible bad conformer Id for molecule {smiles}")
    except ValueError as e:
        logger.error(f"Bad conformer Id for molecule {smiles}. Error: {e}")
    except Exception as e:
        logger.error(f"Could not convert molecule {smiles}. Error: {e}")


def export_to_gfj(smiles_file):
    """ Loads SMILES in the ./smiles.smi file and creates a .gjf for each one of them """
    f = open(smiles_file, 'r')
    smiles = f.readlines()
    f.close()

    for s in tqdm(smiles):
        create_gjf_for_molecule(s)


if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    date = datetime.now()
    # file_log_handler = logging.FileHandler(filename=date.strftime('reactions_%d_%m_%Y_%H_%M.log'), mode='w')
    # logger.addHandler(file_log_handler)

    logger.info('Started')

    export_to_gfj(sys.argv[1])

    logger.info('Finished')