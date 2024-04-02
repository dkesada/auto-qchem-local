import os
import argparse
import sys
import logging
import numpy as np
from datetime import datetime
from rdkit import Chem

from morfeus_ml.morfeus_descriptors import *
from morfeus import read_xyz
from xtb.interface import XTBException


logger = logging.getLogger(__name__)


class MismatchAtomNumber(Exception):
    """ Raised when the number of atoms in a SMILES code does not match the number of atoms in a .xyz file"""
    def __init__(self, smi_n_atoms, xyz_n_atoms):
        super().__init__()
        self.smi_n_atoms = smi_n_atoms
        self.xyz_n_atoms = xyz_n_atoms


class ValidationError(Exception):
    pass


def fix_xyz_file(file_name):
    """ Fix a .xyz file that does not begin with the number of atoms in the molecule """
    print(file_name)
    xyz_file = open(file_name, 'r')
    xyz = xyz_file.readlines()
    xyz_file.close()

    # Discard initial empty lines
    while xyz[0] == '\n':
        xyz = xyz[1:]

    # If the first line is not the number of molecules, calculate it and write it
    if not xyz[0].replace('\n', '').isalnum():
        n = 0
        while n < len(xyz) and xyz[n] != '\n':
            n += 1

        xyz.insert(0, f'{n}\n')
        xyz.insert(1, '\n')

        with open(file_name, 'w') as outfile:
            outfile.write(''.join(l for l in xyz))


def fix_all_xyz(data_dir='./'):
    """ Fix the format of all .xyz files inside a folder and all of its child directories"""
    # Search each directory for .xyz files
    dirs = next(os.walk(data_dir))

    # Base case, no more directories
    if len(dirs[1]) == 0:
        # Get the .xyz files in this folder
        names = np.array(dirs[2])
        names = names[list(map(lambda x: x.endswith('.xyz'), names))]

        for n in names:
            fix_xyz_file(n)

    # Recursive case, further directories
    else:
        prev_dir = os.getcwd()
        for d in dirs[1]:
            # Change the wd to the subdirectory
            os.chdir(f'{prev_dir}/{d}')
            fix_all_xyz('./')
        os.chdir(prev_dir)


def extract_properties_compound(file_name, output_path, n_confs, solvent):
    """
    Extract the morfeus properties of a .smi file, and additionally use a conformer in a .xyz file with the same name.
    The results are written into a .csv file.
    This function can raise an InvalidSmiles exception if the SMILES code cannot be transformed into a rdkit mol and a
    MismatchAtomNumber if the number of atoms in the SMILES and the xyz does not match.
    """
    # Read the .smi file
    smi_file = open(f'{file_name}.smi', 'r')
    smiles = smi_file.readlines()  # This should only contain a single smile
    smi_file.close()
    smiles = smiles[0]
    smiles = smiles.replace('\n', '')

    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise InvalidSmiles

    smi_n_atom = Chem.AddHs(mol).GetNumAtoms()

    try:
        # Read the .xyz file
        elements, coords = read_xyz(f'{file_name}.xyz')
        if len(elements) != Chem.AddHs(mol).GetNumAtoms():
            raise MismatchAtomNumber(smi_n_atom, len(elements))
        conf = Conformer(elements, coords)

        # Create the conformer ensemble
        ce = compute_with_xyz(smiles, conf, n_confs=n_confs, solvent=solvent)
    except FileNotFoundError:
        logging.info(f'No .xyz file found for {file_name}. Computing only with .smi')
        ce = compute(smiles, n_confs=n_confs, solvent=solvent)

    # Calculate the descriptors of the ensemble
    descs = get_descriptors(ce)

    os.makedirs(os.path.dirname(f"{output_path}/{file_name}.csv"), exist_ok=True)
    with open(f"{output_path}/{file_name}.csv", "wb") as f:
        descs.to_frame(smiles).T.to_csv(f, index_label="smiles")


def compute_files(data_dir='./', output_path='jobs', n_confs=5, solvent=None):
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
        logger.info(f'Now processing file {n}')
        tries = 0
        while tries < 10:  # There can be single point calculation errors that require reruns
            try:
                extract_properties_compound(n, output_path, n_confs, solvent)
                break
            except InvalidSmiles:
                logger.warning(f'Could not convert molecule smiles from file {n}.smi')
                break
            except MismatchAtomNumber as e:
                logger.warning(f'Number of atoms does not match in both .smi ({e.smi_n_atoms} atoms) and .xyz '
                               f'({e.xyz_n_atoms}) files. Possible erroneous SMILE code')
                break
            except ValidationError as e:
                logger.warning(f'Problems converting molecule with rdkit. Validation error: {e}')
                break
            except IndexError as e:
                logger.warning(f'Possible problems with the .xyz file and conversion. Index error: {e}')
                break
            except XTBException as e:
                logger.warning(f'Single point calculation failed, retrying ({e})')
                tries += 1
            except Exception as e:
                logger.warning(f'Unexpected exception: {e}')
                break

    # Recursive case, further directories
    if len(dirs[1]) > 0:
        prev_dir = os.getcwd()
        for d in dirs[1]:
            compute_files(f'{prev_dir}/{d}', output_path, n_confs, solvent)
        os.chdir(prev_dir)



if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    date = datetime.now()
    file_handler = logging.FileHandler(filename=date.strftime('morfeus_%d_%m_%Y_%H_%M.log'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description='Compute Conformational Ensemble and its Features')
    parser.add_argument("--data_dir", type=str, help="Directory where to search for .smi files", default='./')
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

    compute_files(data_dir=args.data_dir, output_path=args.output_path, n_confs=args.n_confs, solvent=args.solvent)

    logger.info('Finished')
