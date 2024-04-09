import os
import argparse
import sys
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
import func_timeout

from autoqchem.molecule import molecule
from autoqchem.gaussian_input_generator import gaussian_input_generator


logger = logging.getLogger(__name__)


class AutoChem:
    """
    Class that controls the available auto-qchem operators.
    """

    def __init__(self, log=None, conv_timeout=40, workflow_type="custom", workdir_gjf='./output_gjf', theory="B3LYP",
                 solvent="None", light_basis_set="6-31+G(d,p)", heavy_basis_set="SDD", generic_basis_set="genecp",
                 max_light_atomic_number=25):
        """
        Initialize the AutoChem object with the defaults for each of the components

        :param conv_timeout: seconds until a molecule conversion from SMILES is timed out and deemed failed
        :param workdir_gjf: output directory for the .gjf files
        :param workflow_type: Gaussian workflow type, allowed types are: 'equilibrium' or 'transition_state'
        :param theory: Gaussian supported Functional (e.g., APFD, B3LYP)
        :param solvent: Gaussian supported Solvent (e.g., TETRAHYDROFURAN)
        :param light_basis_set: Gaussian supported basis set for elements up to `max_light_atomic_number` (e.g., 6-31G*, 6-31+G(d,p))
        :param heavy_basis_set: Gaussian supported basis set for elements heavier than `max_light_atomic_number` (e.g., LANL2DZ)
        :param generic_basis_set: Gaussian supported basis set for generic elements (e.g., gencep)
        :param max_light_atomic_number: maximum atomic number for light elements
        """
        if log:
            self.logger = log
        else:
            self.logger = logger

        self.conv_timeout = conv_timeout
        self.workflow_type = workflow_type
        self.workdir_gjf = workdir_gjf
        self.theory = theory
        self.solvent = solvent
        self.light_basis_set = light_basis_set
        self.heavy_basis_set = heavy_basis_set
        self.generic_basis_set = generic_basis_set
        self.max_light_atomic_number = max_light_atomic_number

    def create_gjf_for_molecule(self, smiles):
        """
        Generate Gaussian input files for a molecule

        :param smiles: a SMILES code from a molecule
        """

        # create Gaussian files
        try:
            m = func_timeout.func_timeout(self.conv_timeout, lambda: molecule(smiles, num_conf=1))
            simplified_name = ''.join(e for e in smiles if e.isalnum())
            molecule_workdir = os.path.join(f"{self.workdir_gjf}", simplified_name)
            gig = gaussian_input_generator(m, self.workflow_type, self.workdir_gjf, self.theory, self.solvent,
                                           self.light_basis_set, self.heavy_basis_set, self.generic_basis_set,
                                           self.max_light_atomic_number)
            gig.create_gaussian_files()
            with open(f'output_gjf/{m.inchikey}_conf_0.gjf.smi', "w") as f:
                f.write(smiles)

        except func_timeout.FunctionTimedOut:
            logger.error(f"Timed out! Possible bad conformer Id for molecule {smiles}")
        except ValueError as e:
            logger.error(f"Bad conformer Id for molecule {smiles}. Error: {e}")
        except Exception as e:
            logger.error(f"Could not convert molecule {smiles}. Error: {e}")

    def export_to_gfj(self, smiles_file):
        """ Loads SMILES in the ./smiles.smi file and creates a .gjf for each one of them """
        f = open(smiles_file, 'r')
        smiles = f.readlines()
        f.close()

        for s in tqdm(smiles):
            self.create_gjf_for_molecule(s)

    def compute_csv_files(self, data_dir='./'):
        """
        Recursive function that join the morfeus properties inside separate .csv files into a single one.
        This should only be run when no other .csv files can be found in the folder and subdirectories
        :param data_dir: path to the working directory
        :return: pandas data.frame with all found .csv files
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
                res_rec = self.compute_csv_files(f'{prev_dir}/{d}')
                res = pd.concat([res, res_rec], axis=0)

            os.chdir(prev_dir)

        return res

    def join_csv_files(self, log_dir='./log_values.csv', morfeus_dir='./morfeus_values.csv'):
        """
        Function that joins the log extracted properties and the morfeus properties inside separate .csv files into a single
        one. They will be joined by the file_name column, which contains the original names of the .log and .smi files.
        Both of these should have the same name for this join to work.
        """
        # Load both .csv files
        df_log = pd.read_csv(log_dir)
        df_morfeus = pd.read_csv(morfeus_dir)

        # Join them by the 'file_name' column
        res = pd.merge(df_log, df_morfeus, on='file_name')

        return res

    def generate_dataset(self):
        """
        Function that takes all .log and .csv files in a directory and its subdirectories and creates a full
        .csv dataset with all the information.
        """
        pass
