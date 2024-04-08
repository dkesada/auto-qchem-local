import os
import numpy as np
import itertools

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from autoqchem.molecule import GetSymbol


def extract_from_rdmol(mol: Chem.Mol) -> tuple:
    """Extract information from RDKit Mol object with conformers.

    :param mol: rdkit molecule
    :type mol: rdkit.Chem.Mol
    :return: tuple(elements, conformer_coordinates, connectivity_matrix, charges)
    """

    # Take out elements, coordinates and connectivity matrix
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    conformer_coordinates = []
    for conformer in mol.GetConformers():
        coordinates = conformer.GetPositions()
        conformer_coordinates.append(coordinates)
    conformer_coordinates = np.array(conformer_coordinates)

    connectivity_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)

    return elements, conformer_coordinates, connectivity_matrix, charges


def get_rdkit_mol(elements: list, conformer_coordinates: np.ndarray,
                  connectivity_matrix: np.ndarray, charges: np.ndarray) -> Chem.Mol:
    """Construct an rdkit molecule from elements, positions, connectivity matrix and charges.

    :param elements: list of elements
    :type elements: list
    :param conformer_coordinates: list of conformers 3D coordinate arrays
    :type conformer_coordinates: np.ndarray
    :param connectivity_matrix:  connectivity matrix
    :type connectivity_matrix: np.ndarray
    :param charges: list of formal charges for each element
    :type charges: np.ndarray
    :return: rdkit.Chem.Mol
    """
    _RDKIT_BOND_TYPES = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
        5.0: Chem.BondType.QUINTUPLE,
        6.0: Chem.BondType.HEXTUPLE,
    }

    mol = Chem.RWMol()

    # Add atoms
    for element, charge in zip(elements, charges):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    # Add bonds
    for i, j in zip(*np.tril_indices_from(connectivity_matrix)):
        if i != j:
            bo = connectivity_matrix[i, j]
            if bo != 0:
                bond_type = _RDKIT_BOND_TYPES[float(bo)]
                mol.AddBond(int(i), int(j), bond_type)

    # Add conformers
    add_conformers_to_rdmol(mol, conformer_coordinates)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    return mol


def add_conformers_to_rdmol(mol: Chem.Mol, conformer_coordinates: np.ndarray) -> None:
    """Add conformers to RDKit Mol object.

    :param mol: rdkit mol object
    :type mol: rdkit.Chem.Mol
    :param conformer_coordinates: list of conformers 3D coordinate arrays
    :type conformer_coordinates: np.ndarray
    """

    conformer_coordinates = np.array(conformer_coordinates)
    if len(conformer_coordinates.shape) == 2:
        conformer_coordinates.reshape(-1, conformer_coordinates.shape[0], 3)

    for coordinates in conformer_coordinates:
        conformer = Chem.Conformer()
        for i, coord in enumerate(coordinates):
            point = Geometry.Point3D(*coord)
            conformer.SetAtomPosition(i, point)
        mol.AddConformer(conformer, assignId=True)


def generate_conformations_from_rdkit(smiles, num_conf, rdkit_ff, large_mol) -> tuple:
    """Generate conformational ensemble using rdkit

    :param smiles: SMILES string
    :type smiles: str
    :param num_conf: maximum number of conformations to generate
    :type num_conf: int
    :param rdkit_ff: force field that is supported by rdkit (MMFF94, MMFF94s, UFF)
    :type rdkit_ff: str
    :param large_mol: if true, use random coordinates for conformer generation
    :type large_mol: bool
    :return: tuple(elements, conformer_coordinates, connectivity_matrix, charges)
    """

    # get threads
    n_threads = os.cpu_count() - 1

    # initialize rdmol
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    params = AllChem.EmbedParameters()
    params.useSymmetryForPruning = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.ETversion = 2
    params.pruneRmsThresh = 0.35
    params.numThreads = n_threads
    if large_mol:
        params.useRandomCoords = True

    # embed and optimized conformers
    AllChem.EmbedMultipleConfs(rdmol, num_conf, params)
    if rdkit_ff == "MMFF94":
        AllChem.MMFFOptimizeMoleculeConfs(rdmol, mmffVariant="MMFF94", numThreads=n_threads)
    elif rdkit_ff == "MMFF94s":
        AllChem.MMFFOptimizeMoleculeConfs(rdmol, mmffVariant="MMFF94s", numThreads=n_threads)
    elif rdkit_ff == "UFF":
        AllChem.UFFOptimizeMoleculeConfs(rdmol, numThreads=n_threads)

    elements, conformer_coordinates, connectivity_matrix, charges = extract_from_rdmol(rdmol)

    return elements, conformer_coordinates, connectivity_matrix, charges


def get_light_and_heavy_elements(mol: Chem.Mol, max_light_atomic_number: int) -> tuple:
    """Extract light and heavy elements present in the molecule.

    :param mol: rdkit molecule
    :type mol: rdkit.Chem.Mol
    :param max_light_atomic_number: atomic number of the heaviest element to be treated with light_basis_set
    :type max_light_atomic_number: int
    :return: tuple(light_elements, heavy_elements
    """

    atomic_nums = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
    light_elements = [GetSymbol(n) for n in atomic_nums if n <= max_light_atomic_number]
    heavy_elements = [GetSymbol(n) for n in atomic_nums if n > max_light_atomic_number]
    return light_elements, heavy_elements


def get_rmsd_rdkit(rdmol) -> np.ndarray:
    """Calculate RMSD row-wise with RDKit. This is a computationally slower version, but takes symmetry into account (explores permutations of atom order).

    :param rdmol: rdkit molecule
    :type rdmol: rdkit.Chem.Mol
    :return: np.ndarray
    """

    N = rdmol.GetNumConformers()
    rmsds = np.zeros(shape=(N, N))
    for i, j in itertools.combinations(range(N), 2):
        rms = AllChem.GetBestRMS(rdmol, rdmol, i, j)
        rmsds[i][j] = rms
        rmsds[j][i] = rms

    return rmsds


def prune_rmsds(rdmol, thres):
    """Get a list of conformer indices to keep

    :param rdmol: rdkit molecule
    :type rdmol: rdkit.Chem.Mol
    :param thres: RMSD threshold below which conformers are considered identical (in Angstrom)
    :type thres: float
    :return: list(indices of conformers to keep)
    """

    rmsds = get_rmsd_rdkit(rdmol)

    working_array = rmsds
    candidates = np.array(range(rdmol.GetNumConformers()))

    keep_list = []
    while len(working_array) > 0:
        keeper = candidates[0]
        keep_list.append(keeper)
        rmsd = working_array[0]
        mask = rmsd > thres
        candidates = candidates[mask]
        working_array = working_array[mask, :][:, mask]

    return keep_list
