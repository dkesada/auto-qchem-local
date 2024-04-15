import os
import argparse
import sys
import logging
import numpy as np
import pandas as pd


from api.gjf_generator import GjfGenerator
from api.morfeus_generator import MorfeusGenerator


logger = logging.getLogger(__name__)


class AutoChem:
    """
    Class that controls the available auto-qchem operators.
    """

    def __init__(self, log=None, conv_timeout=40, workflow_type="custom", workdir_gjf='./output_gjf', theory="B3LYP",
                 solvent="None", light_basis_set="6-31+G(d,p)", heavy_basis_set="SDD", generic_basis_set="genecp",
                 max_light_atomic_number=25, n_confs=5):
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
        self.gjf_gen = GjfGenerator(log=self.logger, conv_timeout=conv_timeout, workflow_type=workflow_type,
                                    workdir_gjf=workdir_gjf, theory=theory, solvent=solvent,
                                    light_basis_set=light_basis_set, heavy_basis_set=heavy_basis_set,
                                    generic_basis_set=generic_basis_set,
                                    max_light_atomic_number=max_light_atomic_number)

        self.morf_gen = MorfeusGenerator(log=self.logger, n_confs=n_confs, solvent=solvent)

    def generate_gjf_files(self, smiles_file):
        """Generate gjf files for all smiles inside the provided .smi file"""
        pass

    def compute_morfeus(self, dir):
        """ Compute the morfeus descriptors for each .smi file found and save them as .csv files"""
        pass

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
