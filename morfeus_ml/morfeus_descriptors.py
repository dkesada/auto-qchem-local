import os
import argparse
import pickle

import pandas as pd
import numpy as np

from rdkit import Chem

from morfeus.conformer import ConformerEnsemble, Conformer
from morfeus import Sterimol, BuriedVolume, XTB, ConeAngle, SASA, Dispersion, Pyramidalization, SolidAngle, BiteAngle
from morfeus.data import atomic_symbols, atomic_numbers
from morfeus_ml.data import metals
from morfeus_ml.geometry import get_closest_atom_to_metal, get_central_carbon

import openbabel.pybel as pybel
from openbabel.openbabel import OBMol
from math import isnan


# constants
os.environ['NWCHEM_BASIS_LIBRARY'] = '/home/zuranski/miniconda3/envs/qml/share/nwchem/libraries/'


class InvalidSmiles(Exception):
    """ Raised when a SMILES cannot be converted into either a rdkit or openbabel mol """
    pass


def convert_to_symbol(elements):
    """ Converts an element vector from number format to str format """
    return np.array(list(map(lambda x: atomic_symbols[x], elements)))


def convert_to_numbers(elements):
    """ Converts an element vector from str format to number format """
    return np.array(list(map(lambda x: atomic_numbers[x], elements)))


def convert_to_ob_mol(smiles, n_confs):
    """ Attempt to convert a smiles into an openbabel molecule with the different forcefields available """
    coords_nan = True

    py_mol = pybel.readstring("smi", smiles)
    py_mol.make3D(forcefield='mmff94')
    ce = ConformerEnsemble.from_ob_ga(py_mol.OBMol, num_conformers=n_confs)

    if isnan(sum(sum(ce[0].coordinates))):  # If there are nans in the coordinates, try another forcefield
        py_mol = pybel.readstring("smi", smiles)
        py_mol.make3D(forcefield='uff')  # The default forcefield can generate NaNs on non-convertible rdkit smiles
        ce = ConformerEnsemble.from_ob_ga(py_mol.OBMol, num_conformers=n_confs)

    if isnan(sum(sum(ce[0].coordinates))):
        py_mol = pybel.readstring("smi", smiles)
        py_mol.make3D(forcefield='ghemical')  # The default forcefield can generate NaNs on non-convertible rdkit smiles
        ce = ConformerEnsemble.from_ob_ga(py_mol.OBMol, num_conformers=n_confs)

    if isnan(sum(sum(ce[0].coordinates))):  # If no conversion works, raise an exception
        raise InvalidSmiles('Could not convert SMILES with either rdkit or openbabel')

    return ce


def get_specific_atom_sterimol(elements, coords, metal_idx):
    """
    We have 3 different cases: metal-phosphorus, nitrogen-metal-nitrogen and metal-(nitrogen-carbon-nitrogen)
    In each case, we want different indexes for the sterimol: the phosphorus index, the closer nitrogen index and
    the carbon between the nitrogens index
    """
    res = None

    # Standardize the str format
    if not isinstance(elements[0], str):
        elements = convert_to_symbol(elements)

    # The case of the phosphorus, I get the closest one to the metal
    if 'P' in elements:
        res = get_closest_atom_to_metal('P', elements, metal_idx, coords)
    # The case of the metal-(nitrogen-carbon-nitrogen), I get the carbon in the middle
    else:
        res = get_central_carbon(elements, coords, metal_idx)

    return res


def reorder_coords(coords, elems, new_elems):
    """
    Reorders the coordinates of a conformer to have its atoms in the same order provided by new_elems.
    This is necesary when introducing an external conformer from an .xyz file into a conformer ensemble,
    because the ensemble may have different ordering
    """
    # Check for the same format in elems and new_elems
    if isinstance(elems[0], str) and not isinstance(new_elems[0], str):
        new_elems = convert_to_symbol(new_elems)
    elif isinstance(new_elems[0], str) and not isinstance(elems[0], str):
        elems = convert_to_symbol(elems)

    # Move each coordinate atom by atom
    new_coords = coords.copy()
    idx = 0
    for e in new_elems:
        # Search the first corresponding atom and move it
        idx_old = np.where(e == elems)[0][0]
        new_coords[idx] = coords[idx_old]
        # Delete that atom from the old coords
        elems = np.delete(elems, idx_old)
        coords = np.delete(coords, idx_old, 0)
        idx += 1

    return new_coords


def compute(smiles, n_confs=None, program='xtb', method='GFN2-xTB', basis=None, solvent=None):
    """Calculates a conformer ensemble using rdkit, then optimizes
    each conformer geometry using GFN2-xTB and calculates their properties"""

    # create conformer ensemble
    ce = ConformerEnsemble.from_rdkit(smiles, n_confs=n_confs,
                                      n_threads=os.cpu_count() - 1)
    # except Exception:  # Rdkit conversion fails for some SMILES, using openbabel instead. Can't get specific exception
    #     ce = convert_to_ob_mol(smiles, n_confs)

    # To avoid inconsistent charge and multiplicity
    ce.charge = 0
    ce.multiplicity = 1

    # optimize conformer geometries
    ce.optimize_qc_engine(program=program,
                          model={'method': method, "basis": basis, "solvent": solvent},
                          procedure="geometric")
    ce.prune_rmsd()

    # compute energies of the single point calculations
    ce.sp_qc_engine(program=program, model={'method': method, "basis": basis, "solvent": solvent})

    # sort on energy and generate an rdkit molecule
    ce.sort()
    ce.generate_mol()

    # compute xtb for the conformers for descriptor calculations
    for conf in ce:
        conf.xtb = XTB(ce.elements, conf.coordinates, solvent=solvent, version='2')

    return ce


def compute_with_xyz(smiles, xyz_conf, n_confs=None, program='xtb', method='GFN2-xTB', basis=None, solvent=None):
    """Calculates a conformer ensemble using rdkit, then optimizes
    each conformer geometry using GFN2-xTB and calculates their properties. Additionally, this function loads a
    previous conformer in an .xyz file and adds it to the ensemble. Usually, this conformer is very close to the
    minimal energy and already optimized. """

    # create conformer ensemble
    ce = ConformerEnsemble.from_rdkit(smiles, n_confs=n_confs,
                                      n_threads=os.cpu_count() - 1)
    # except Exception:  # Rdkit conversion fails for some SMILES, using openbabel instead. Can't get specific exception
    #    ce = convert_to_ob_mol(smiles, n_confs)

    # To avoid inconsistent charge and multiplicity
    ce.charge = 0
    ce.multiplicity = 1

    # optimize conformer geometries
    ce.optimize_qc_engine(program=program,
                          model={'method': method, "basis": basis, "solvent": solvent},
                          procedure="geometric")
    ce.prune_rmsd()

    # add the xyz conformer to the ensemble
    new_coords = reorder_coords(xyz_conf.coordinates, xyz_conf.elements, ce.elements)
    ce.add_conformers([new_coords])

    # compute energies of the single point calculations
    ce.sp_qc_engine(program=program, model={'method': method, "basis": basis, "solvent": solvent})

    # sort on energy and generate an rdkit molecule
    ce.sort()
    ce.generate_mol()

    # compute xtb for the conformers for descriptor calculations
    for conf in ce:
        conf.xtb = XTB(ce.elements, conf.coordinates, solvent=solvent, version='2')

    return ce


def get_descriptors(conf_ensemble, substructure=None, substructure_labels=None,
                    sterimol_pairs=[]):
    """Extract descriptors from the conformational ensemble"""

    def get_substructure_match(conf_ensemble, substructure):
        """Helper function to find substructure match"""

        sub = Chem.MolFromSmarts(substructure)
        matches = conf_ensemble.mol.GetSubstructMatches(sub)
        if len(matches) > 1:
            raise ValueError(f"Substructer {substructure} is matched more than once in the molecule.")
        elif len(matches) == 0:
            raise ValueError(f"Substructer {substructure} not found in the molecule.")
        else:
            match = [m + 1 for m in matches[0]]  # add one to conform with morfeus 1-indexing vs rdkit 0-indexing

        print(match)
        return match

    def make_substructure_labels(match, substructure_labels):
        """Helper function to assign substructre atom labels"""

        if substructure_labels is None:
            labels = [f"atom{i}" for i in range(len(match))]
        else:
            if len(substructure_labels) != len(match):
                raise ValueError(f"Length of labels ({len(substructure_labels)}) is different\
                than the lenght of the substructure {substructure} match ({len(match)})")
            else:
                labels = substructure_labels

        return labels

        # prep for the substructure

    if substructure is not None:
        match = get_substructure_match(conf_ensemble, substructure)
        labels = make_substructure_labels(match, substructure_labels)
    else:
        match = None
    
    conf_idx = 0
    # loop over the conformers and get properties
    for conf in conf_ensemble.conformers:

        # Get conformer properties
        get_conf_props(conf)

        if match is not None:
            # atomic features
            charges = conf.xtb.get_charges()
            electro = conf.xtb.get_fukui('electrophilicity')
            nucleo = conf.xtb.get_fukui('nucleophilicity')
            for idx, label in zip(match, labels):
                # charge, electrophilicity, nucleophilicity
                conf.properties[f"{label}_charge"] = charges[idx]
                conf.properties[f"{label}_electro"] = electro[idx]
                conf.properties[f"{label}_nucleo"] = nucleo[idx]

                # buried volumes
                conf.properties[f"{label}_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                                idx).fraction_buried_volume

            # Sterimols
            for pair in sterimol_pairs:
                i1, i2 = pair[0], pair[1]
                match1, match2 = match[pair[0]], match[pair[1]]
                label = f"{labels[i1]}{labels[i2]}"
                s = Sterimol(conf_ensemble.elements, conf.coordinates, match1, match2)

                conf.properties[f"{label}_L"] = s.L_value
                conf.properties[f"{label}_B1"] = s.B_1_value
                conf.properties[f"{label}_B5"] = s.B_5_value
                conf.properties[f"{label}_length"] = s.bond_length

    props = {}
    for key in conf_ensemble.get_properties().keys():
        props[f"{key}_Boltz"] = conf_ensemble.boltzmann_statistic(key)
        props[f"{key}_Emin"] = conf_ensemble.get_properties()[key][0]
    
    return pd.Series(props)


def get_conf_props(conf, elements=None, coordinates=None, solvent=None):
    """
    Get the properties of a conformer. If the conformer is None, then one will be created with the provided elements,
    coordinates and solvent.
    """
    if not conf:
        conf = Conformer(elements, coordinates)
        conf.xtb = XTB(elements, coordinates, solvent=solvent, version='2')

    # molecular features
    conf.properties['IP'] = conf.xtb.get_ip(corrected=True)
    conf.properties['EA'] = conf.xtb.get_ea()
    conf.properties['HOMO'] = conf.xtb.get_homo()
    conf.properties['LUMO'] = conf.xtb.get_lumo()

    dip = conf.xtb.get_dipole()
    conf.properties['dipole'] = np.sqrt(dip.dot(dip))

    conf.properties['electro'] = conf.xtb.get_global_descriptor("electrophilicity", corrected=True)
    conf.properties['nucleo'] = conf.xtb.get_global_descriptor("nucleophilicity", corrected=True)

    # Global XTB
    conf.properties["electro"] = conf.xtb.get_global_descriptor("electrophilicity", corrected=True)
    conf.properties["nucleo"] = conf.xtb.get_global_descriptor("nucleophilicity", corrected=True)
    conf.properties["electrofug"] = conf.xtb.get_global_descriptor("electrofugality", corrected=True)
    conf.properties["nucleofug"] = conf.xtb.get_global_descriptor("nucleofugality", corrected=True)

    # SASA
    sasa = SASA(conf.elements, conf.coordinates)
    conf.properties["SASA_area"] = sasa.area
    conf.properties["SASA_vol"] = sasa.volume

    # Dispersion
    disp = Dispersion(conf.elements, conf.coordinates)
    conf.properties["Dispersion_area"] = disp.area
    conf.properties["Dispersion_vol"] = disp.volume
    conf.properties["Dispersion_P_int"] = disp.p_int

    # Metal part

    metallic = False
    # Check if there's a metal in the configuration
    idx = [x for x in range(len(conf.elements)) if conf.elements[x] in metals]
    if len(idx) > 0:
        metallic = True
        metal_idx = idx[0] + 1  # The index has to be 1-indexed for morfeus

    # Buried volumes (VBur)
    if metallic:
        bv = BuriedVolume(conf.elements, conf.coordinates, metal_idx)
        bv.octant_analysis()
        conf.properties["VBur"] = bv.buried_volume
        conf.properties["Free_VBur"] = bv.free_volume
        bv.compute_distal_volume(method="sasa")
        conf.properties["Sasa_distal_VBur"] = bv.distal_volume
        conf.properties["Sasa_mol_VBur"] = bv.molecular_volume
        bv.compute_distal_volume(method="buried_volume")
        conf.properties["Buried_distal_VBur"] = bv.distal_volume
        conf.properties["Buried_mol_VBur"] = bv.molecular_volume

    # Pyramid
    if metallic:
        pyr = Pyramidalization(conf.coordinates, metal_idx)
        conf.properties["Pyr_alpha"] = pyr.alpha
        conf.properties["Pyr_P"] = pyr.P
        conf.properties["Pyr_P_angle"] = pyr.P_angle

    # Bite angle
    # if metallic:
    #     bite = BiteAngle(conf.coordinates, metal_idx, metal_idx-1, metal_idx+1)  # Assuming that the other atoms are before and after the metal...
    #     conf.properties["Bite_angle"] = bite.angle

    # Solid angle
    if metallic:
        solid = SolidAngle(conf.elements, conf.coordinates, metal_idx)
        conf.properties["Cone_angle"] = solid.cone_angle
        conf.properties["Solid_angle"] = solid.solid_angle
        conf.properties["Solid_angle_G"] = solid.G

    # Cone angle
    if metallic:
        cone_angle = ConeAngle(conf.elements, conf.coordinates, metal_idx)
        conf.properties["Cone_angle"] = cone_angle.cone_angle
        conf.properties["Cone_angle_NAtoms"] = len(cone_angle.tangent_atoms)

    # Sterimol
    if metallic:
        idx = get_specific_atom_sterimol(conf.elements, conf.coordinates, metal_idx-1)
        sterimol = Sterimol(conf.elements, conf.coordinates, idx+1, metal_idx)
        conf.properties["Sterimol_L"] = sterimol.L_value
        conf.properties["Sterimol_B_1"] = sterimol.B_1_value
        conf.properties["Sterimol_B_5"] = sterimol.B_5_value
        sterimol.bury(method="delete")
        conf.properties["Sterimol_bur_L"] = sterimol.L_value
        conf.properties["Sterimol_bur_B_1"] = sterimol.B_1_value
        conf.properties["Sterimol_bur_B_5"] = sterimol.B_5_value



if __name__ == "__main__":
    """executable script"""

    parser = argparse.ArgumentParser(description='Compute Conformational Ensemble and its Features')
    parser.add_argument("smiles", type=str, help="SMILES string of the molecule")
    parser.add_argument("name", type=str, help="Name of the molecule")
    parser.add_argument("output_path", type=str, help="Storage output path")
    parser.add_argument("--n_confs", type=int, help="Optional number of conformers to initially generate with RDKit. "
                                                    "If not specified a default is used based on the number of"
                                                    " rotatable bonds (50 if n_rot <=7, 200 if n_rot <=12, "
                                                    "300 for larger n_rot)",
                        default=None)
    parser.add_argument("--solvent", type=str, help="XTB supported solvents and their names can be found at "
                                                    "https://xtb-docs.readthedocs.io/en/latest/gbsa.html",
                        default=None)
    parser.add_argument("--descriptors", action='store_true',
                        help="Compute molecular descriptors")
    parser.add_argument("--substructure", type=str,
                        help="Substructure atoms get their individual descriptors")
    parser.add_argument("--substructure_labels", type=str,
                        nargs='+', help='List of labels for substructre query atoms', default=None)


    def sterimol_pair(s):
        try:
            i1, i2 = map(int, s.split(","))
            return i1, i2
        except:
            raise argparse.ArgumentTypeError("Sterimol pairs must be index1, index2")


    parser.add_argument("--sterimol_pairs", type=sterimol_pair, nargs="+",
                        help='Pair of atoms from the substructure match to extract the sterimol parameters (0-indexed)')

    args = parser.parse_args()
    print(args.__dict__)

    # compute the conformational ensemble
    args.smiles = "Cl[Cu][P](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3"
    m = compute(args.smiles, n_confs=args.n_confs, solvent=args.solvent)
    print(m.get_energies())

    if args.descriptors:
        descs = get_descriptors(m, substructure=args.substructure,
                                substructure_labels=args.substructure_labels,
                                sterimol_pairs=args.sterimol_pairs)

        with open(f"{args.output_path}/{args.name}_descriptors.csv", "wb") as f:
            descs.to_frame(args.smiles).T.to_csv(f, index_label="smiles")
    
    print(m)
    print(f)
    # save output
    #with open(f"{args.output_path}/{args.name}.pkl", "wb") as f:
    #    pickle.dump(m, f)
