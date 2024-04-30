import os
from datetime import datetime
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm


from autoqchem_local.api.gjf_generator import GjfGenerator
from autoqchem_local.api.morfeus_generator import MorfeusGenerator, MismatchAtomNumber
from autoqchem_local.api.log_extractor import LogExtractor
from autoqchem_local.morfeus_ml.morfeus_descriptors import InvalidSmiles

logger = logging.getLogger(__name__)

try:
    from xtb.interface import XTBException
except ImportError as e:
    logger.warning(f'No version found for xtb. Windows machines are not compatible. Exception: {e}')


class AutoChem:
    """
    Class that controls the available auto-qchem local operators. Specific properties for each of the generator
    elements are defined when creating this AutoChem class in the __init__().

    There are 6 relevant entry points to the api:

    - generate_gjf_files(smiles_file='./smiles.smi'): this function generates the input .gjf files for Gaussian and
        the .smi files for morfeus calculation for each SMILES found in smiles_file. It is usually the first step
        if Gaussian is used.

    - process_morfeus(data_dir='./', output_path='jobs'): this function calculates the morfeus properties of all .smi
        files found in data_dir (ideally, this .smi files are the ones generated with generate_gjf_files()). All
        properties are then stored as separate .csv files. Multiple files are generated in case there is some problem
        with the machine making the calculations, so that if it somehow fails or shutdowns, calculated molecules will
        not be lost.

    - join_morfeus_csv_files(data_dir='./'): this function will join all .csv files found in a directory into
        a single .csv file. It should be run after process_morfeus() has finished all calculations, so that a single
        dataset with the calculated properties for all compounds is generated.

    - process_log_files(data_dir='./', output_path='./'): this function extracts the relevant information from
        all Gaussian .log output files found into a single .csv file with all the information.

    - join_log_and_morfeus(log_dir='./log_values.csv', morfeus_dir='./morfeus_values.csv'): this function merges the
        .csv files obtained from morfeus and the .log files into a single dataset

    - generate_dataset(data_dir='./', gaussian=True): in case one wants to run the full pipeline and not each individual
        part, this function runs all morfeus calculations and log extractions and returns a complete dataset. The only
        part not covered is the gjf generation and the Gaussian execution. This can be run with or without .log files
        from Gaussian.

    """

    def __init__(self, log=None, log_to_file=True, conv_timeout=40, workflow_type="custom", workdir_gjf='./output_gjf',
                 theory="B3LYP", solvent=None, light_basis_set="6-31+G(d,p)", heavy_basis_set="SDD",
                 generic_basis_set="genecp", max_light_atomic_number=25, n_confs=5):
        """
        Initialize the AutoChem object with the defaults for each of the components

        :param conv_timeout: seconds until a molecule conversion from SMILES is timed out and deemed failed
        :param workdir_gjf: output directory for the .gjf files
        :param workflow_type: Gaussian workflow type, allowed types are: 'equilibrium' or 'transition_state'
        :param theory: Gaussian supported Functional (e.g., APFD, B3LYP)
        :param solvent: Gaussian supported Solvent (e.g., none, TETRAHYDROFURAN)
        :param light_basis_set: Gaussian supported basis set for elements up to `max_light_atomic_number`
        (e.g., 6-31G*, 6-31+G(d,p))
        :param heavy_basis_set: Gaussian supported basis set for elements heavier than `max_light_atomic_number`
        (e.g., LANL2DZ)
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

        self.log_ext = LogExtractor(log=self.logger)

        # Prepare the logger to output into both console and a file with the desired format
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        if log_to_file:
            date = datetime.now()
            file_handler = logging.FileHandler(filename=date.strftime('api_%d_%m_%Y_%H_%M.log'), mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @staticmethod
    def _format_path(path):
        """ Make sure that the path ends with a '/' symbol """
        if path[-1] != '/':
            path = path + '/'

        return path

    def generate_gjf_files(self, smiles_file='./smiles.smi'):
        """
        Entry point for the generation of gjf files.
        This function loads all SMILES in the smiles_file file and creates a .gjf and .smi files with the same names
        for each one of them inside the workdir_gjf folder provided when creating the api object.
        """
        f = open(smiles_file, 'r')
        smiles = f.readlines()
        f.close()

        for s in tqdm(smiles):
            self.gjf_gen.create_gjf_for_molecule(s)

    def _rec_compute_morfeus(self, data_dir='./', output_path='jobs'):
        """
        Recursive function that will calculate the morfeus properties for all pairs .xyz and .smi files found with
        the same name. If only .smi files are found, they will be used by themselves.
        It will create .csv files with all results found
        """
        # Change the wd to the subdirectory
        os.chdir(data_dir)
        # Search each directory for .smi files
        dirs = next(os.walk('./'))

        # Base case, process the found files or do nothing if there is none and there are no more directories

        # Get the .smi files in this folder
        names = np.array(dirs[2])
        names = names[list(map(lambda x: x.endswith('.smi'), names))]
        names = list(map(lambda x: x[:-4], names))

        for n in names:
            self.logger.info(f'Now processing file {n}')
            tries = 0
            while tries < 10:  # There can be single point calculation errors that require reruns
                try:
                    self.morf_gen.extract_properties_compound(n, output_path)
                    break
                except InvalidSmiles:
                    self.logger.warning(f'Could not convert molecule smiles from file {n}.smi')
                    break
                except MismatchAtomNumber as e:
                    self.logger.warning(f'Number of atoms does not match in both .smi ({e.smi_n_atoms} atoms) and .xyz ({e.xyz_n_atoms} atoms) files. Possible erroneous SMILE code')
                    break
                except IndexError as e:
                    self.logger.warning(f'Possible problems with the .xyz file and conversion. Index error: {e}')
                    break
                except XTBException as e:
                    self.logger.warning(f'Single point calculation failed, retrying ({e})')
                    tries += 1
                except Exception as e:
                    self.logger.warning(f'Unexpected exception: {e}')
                    break

        # Recursive case, further directories
        if len(dirs[1]) > 0:
            prev_dir = os.getcwd()
            for d in dirs[1]:
                self._rec_compute_morfeus(f'{prev_dir}/{d}', output_path)
            os.chdir(prev_dir)

    def process_morfeus(self, data_dir='./', output_path='jobs'):
        """
        Entry point that runs a morfeus calculation for each .smi file found in a directory and its subdirectories.
        This function will process all .smi files found and use morfeus to calculate the molecule properties.
        The .smi files are assumed to have only 1 line with a SMILES code.
        :param data_dir: the path to the directory where to perform the search and processing of .smi files
        :param output_path: the folder to be created in each subdirectory with the processed .csv files
        """
        prev_dir = os.getcwd()
        self._rec_compute_morfeus(data_dir=data_dir, output_path=output_path)
        os.chdir(prev_dir)

    def _rec_join_morfeus_csv_files(self, data_dir='./'):
        """
        Recursive function that join the morfeus properties inside separate .csv files into a single one.
        This should only be run when no other .csv files can be found in the folder and subdirectories
        :param data_dir: path to the working directory
        :return: pandas dataframe with all found .csv files
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
            self.logger.info(f'Now processing file {n}')
            df = pd.read_csv(f'{n}.csv')
            if df.columns[0] != 'smiles':  # A .csv that is not a morfeus output
                self.logger.warning(f'File {n}.csv does not begin with a SMILES column, ignoring possible non-morfeus .csv')
            else:
                df.insert(0, 'file_name', n)  # Insert the file name for joining with log output later
                res = pd.concat([res, df], axis=0)

        # Recursive case, further directories
        if len(dirs[1]) > 0:
            prev_dir = os.getcwd()
            for d in dirs[1]:
                # Change the wd to the subdirectory
                res_rec = self._rec_join_morfeus_csv_files(f'{prev_dir}/{d}')
                res = pd.concat([res, res_rec], axis=0)

            os.chdir(prev_dir)

        return res

    def join_morfeus_csv_files(self, data_dir='./'):
        """
        Entry point for joining the morfeus properties of all molecules from different .csv files into a single one
        """
        prev_dir = os.getcwd()
        res = self._rec_join_morfeus_csv_files(data_dir=data_dir)
        res.reset_index(drop=True, inplace=True)
        res.to_csv('morfeus_values.csv', index=False)
        os.chdir(prev_dir)

    def _rec_compute_log_files(self, data_dir='./'):
        """
        Recursive function that processes each .log file found in a directory and its subdirectories. It will convert
        all .log files found into a single pandas dataframe
        :param data_dir: the directory where to perform the search and processing of .log files
        :return: the processed .log files inside a pandas dataframe
        """
        # Search each directory for .log files
        os.chdir(data_dir)
        dirs = next(os.walk('./'))

        # Base case, process the found files or do nothing if there is none and there are no more directories
        res = pd.DataFrame()

        # Get the .log files in this folder
        names = np.array(dirs[2])
        names = names[list(map(lambda x: x.endswith('.log'), names))]
        names = list(map(lambda x: x[:-4], names))

        for n in names:
            self.logger.info(f'Now processing file {n}')
            try:
                df = self.log_ext.export_to_pandas(f'{n}.log')
                df.insert(0, 'file_name', n)  # Insert the file name for joining with morfeus output later
                if len(df.columns) > len(res.columns):
                    res = pd.concat([df, res], axis=0)
                else:
                    res = pd.concat([res, df], axis=0)
            except AttributeError as e:
                self.logger.warning(f'Possible problem processing the log file: {e}')

        # Recursive case, further directories
        if len(dirs[1]) > 0:
            prev_dir = os.getcwd()
            for d in dirs[1]:
                # Change the wd to the subdirectory
                res_rec = self._rec_compute_log_files(f'{prev_dir}/{d}')
                if len(res_rec.columns) > len(res.columns):
                    res = pd.concat([res_rec, res], axis=0)
                else:
                    res = pd.concat([res, res_rec], axis=0)

            os.chdir(prev_dir)

        return res

    def process_log_files(self, data_dir='./', output_path='./'):
        """
        Entry point for processing all .log files found inside a directory and its subdirectories.
        This function will process all .log files found in data_dir and compile the extracted information
        into a .csv file saved to output_path as log_values.csv
        """
        self.logger.info('Started .log processing')

        prev_dir = os.getcwd()
        res = self._rec_compute_log_files(data_dir=data_dir)
        os.chdir(prev_dir)
        res.reset_index(drop=True, inplace=True)
        res.to_csv(f'{self._format_path(output_path)}log_values.csv', index=False)

        self.logger.info('Finished .log processing')

    def join_log_and_morfeus(self, log_dir='./log_values.csv', morfeus_dir='./morfeus_values.csv'):
        """
        Function that joins the log extracted properties and the morfeus properties inside separate .csv files into a
        single one. They will be joined by the file_name column, which contains the original names of the .log and
        .smi files. Both of these should have the same name for this join to work.
        """
        self.logger.info('Joining .log and morfeus .csv files')

        # Load both .csv files
        df_log = pd.read_csv(log_dir)
        df_morfeus = pd.read_csv(morfeus_dir)

        # Join them by the 'file_name' column
        res = pd.merge(df_log, df_morfeus, on='file_name')

        self.logger.info('Finished joining both files')

        return res

    def generate_dataset(self, data_dir='./', gaussian=True):
        """
        Global entry point of the api. This function runs the whole process from calculating the morfeus properties of
        each .smi file found. It takes all .log and .csv files in data_dir and its subdirectories and creates a full
        .csv dataset with all the information. If the gaussian parameter is set to false, no .log files from
        Gaussian calculation are needed and only morfeus properties will be calculated.
        :param data_dir: the path to the directory where all .smi and .log files are stored. These files can be stored
        in further subdirectories inside
        :param gaussian: whether to process .log files or not
        """
        self.logger.info('Initiating full pipeline')

        # Calculate morfeus properties
        self.process_morfeus(data_dir=data_dir)
        # Join all morfeus .csv separate files into a single one
        self.join_morfeus_csv_files(data_dir=data_dir)

        if gaussian:
            # Extract all .log values
            self.process_log_files(data_dir=data_dir, output_path=data_dir)
            # Merge the morfeus and the log .csv files into a single dataframe
            res = self.join_log_and_morfeus(log_dir=f'{self._format_path(data_dir)}log_values.csv',
                                            morfeus_dir=f'{self._format_path(data_dir)}morfeus_values.csv')
        else:
            res = pd.read_csv(f'{self._format_path(data_dir)}morfeus_values.csv')

        # Store the dataframe as the final .csv file
        res.reset_index(drop=True, inplace=True)
        res.to_csv(f'{self._format_path(data_dir)}full_dataset.csv', index=False)

        self.logger.info('Finished full pipeline')
