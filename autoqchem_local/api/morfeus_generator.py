import os
import logging
from rdkit import Chem
from autoqchem_local.morfeus_ml.morfeus_descriptors import compute, compute_with_xyz, InvalidSmiles, Conformer, get_descriptors
from morfeus import read_xyz
from pandas import DataFrame

logger = logging.getLogger(__name__)


class MismatchAtomNumber(Exception):
    """ Raised when the number of atoms in a SMILES code does not match the number of atoms in a .xyz file"""
    def __init__(self, smi_n_atoms, xyz_n_atoms):
        super().__init__()
        self.smi_n_atoms = smi_n_atoms
        self.xyz_n_atoms = xyz_n_atoms


class MorfeusGenerator:
    def __init__(self, log=None, n_confs=5, solvent=None):
        if log:
            self.logger = log
        else:
            self.logger = logger

        self.n_confs = n_confs
        self.solvent = solvent

    def extract_properties_compound(self, file_name, output_path):
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
            ce = compute_with_xyz(smiles, conf, n_confs=self.n_confs, solvent=self.solvent)
        except FileNotFoundError:
            self.logger.info(f'No .xyz file found for {file_name}. Computing only with .smi')
            ce = compute(smiles, n_confs=self.n_confs, solvent=self.solvent)

        # Calculate the descriptors of the ensemble
        descs = get_descriptors(ce)

        os.makedirs(os.path.dirname(f"{output_path}/{file_name}.csv"), exist_ok=True)
        with open(f"{output_path}/{file_name}.csv", "wb") as f:
            descs.to_frame(smiles).T.to_csv(f, index_label="smiles")
            
    def extract_properties_mol(self, mol):
        """
        Extract the morfeus properties of a rdkit.Chem.rdchem.Mol file and return them as a pandas DataFrame
        """

        ce = compute(mol, n_confs=self.n_confs, solvent=self.solvent)

        # Calculate the descriptors of the ensemble
        descs = get_descriptors(ce)
        descs = pd.DataFrame([descs.values], columns=descs.index)

        return descs
