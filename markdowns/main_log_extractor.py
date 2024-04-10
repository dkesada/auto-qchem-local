# Code partly adapted from https://github.com/doyle-lab-ucla/auto-qchem/blob/master/autoqchem/gaussian_log_extractor.py

import logging
import argparse
from datetime import datetime
import re
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

try:
    from openbabel import pybel  # openbabel 3.0.0

    GetVdwRad = pybel.ob.GetVdwRad
except ImportError:
    import pybel  # openbabel 2.4

    table = pybel.ob.OBElementTable()
    GetVdwRad = table.GetVdwRad


logger = logging.getLogger(__name__)
float_or_int_regex = "[-+]?[0-9]*\.[0-9]+|[0-9]+"


def occupied_volume(geometry_df, atom_idx, r, mesh_density=30) -> float:
    """Compute occupied volume fraction within a sphere of radius 'r' for an atom at position 'atom_idx'. Each atom \
    radius is taken to be its Van der Waals radius.

    :param geometry_df: geometry dataframe, must contain 'X', 'Y', 'Z' and 'AN' (atomic number) columns
    :type geometry_df: pd.DataFrame
    :param atom_idx: index of the atom to use as 'central' atom
    :type atom_idx: int
    :param r: occupied volume radius in Angstroms
    :type r: float
    :param mesh_density: density of the mesh for numerical integration (MAX=100)
    :type mesh_density: int
    :return: float, occupied volume fraction
    """

    # make sure mesh_density is not outrageous
    max_mesh_density = 100
    if mesh_density > max_mesh_density:
        mesh_density = max_mesh_density
        logger.warning(f"Mesh density {mesh_density} is larger than allowed "
                       f"max of {max_mesh_density}. Using {max_mesh_density} instead.")

    # fetch Van der Waals radii for atoms, r
    atom_r = geometry_df['AN'].map(GetVdwRad)

    # isolate coordinates
    coords = geometry_df[list('XYZ')]

    # create cubic mesh, then make it spherical, then move it into central atom
    ticks = np.linspace(-r, r, mesh_density)
    x, y, z = np.meshgrid(ticks, ticks, ticks)
    mesh = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    mesh = mesh[cdist(mesh, np.array([[0., 0., 0.]]), metric='sqeuclidean').ravel() < r ** 2]
    mesh = mesh + coords.iloc[atom_idx].values

    # filter atoms that are certainly not in the mesh, d > R + r
    atom_distances = cdist(coords.iloc[[atom_idx]], coords)[0]
    mesh_overlap_indices = (atom_distances - atom_r) < r

    # compute distance of every atom to every point in the mesh (this is the longest operation)
    distances_sq = pd.DataFrame(cdist(coords[mesh_overlap_indices], mesh, metric='sqeuclidean'),
                                index=atom_r[mesh_overlap_indices].index)
    # mesh cells are occupied if their distances are less then Van der Waals radius
    # the below comparison requires matching indexes in the distances_sq matrix and atom_r series
    occupancy = distances_sq.lt(atom_r[mesh_overlap_indices] ** 2, axis=0)

    # mesh cells are occupied if they are occupied by at least 1 atom
    occupied = occupancy.any()

    return occupied.sum() / mesh.shape[0]


class gaussian_log_extractor(object):
    """"""

    def __init__(self, log_file_path):
        """Initialize the log extractor. Extract molecule geometry and atom labels.

        :param log_file_path: local path of the log file
        :type log_file_path: str
        """

        with open(log_file_path) as f:
            self.log = f.read()
        self.log_file_path = log_file_path  # record keeping; used to output log file path in case of exception

        # initialize descriptors
        self.descriptors = {}
        self.atom_freq_descriptors = None
        self.atom_td_descriptors = None
        self.atom_descriptors = None
        self.vbur = None
        self.modes = None
        self.mode_vectors = None
        self.transitions = None
        self.n_tasks = len(re.findall("Normal termination", self.log))

        self._split_parts()  # split parts

    def check_for_exceptions(self):
        """Go through the log file and look for known exceptions, truncated file, negative frequencies,
        incomplete optimization, and raise a corresponding exception
        """
        try:
            self.get_atom_labels()  # fetch atom labels
            self.get_geometry()  # fetch geometries for each log section
        except IndexError:
            raise NoGeometryException()

        try:
            self._get_frequencies_and_moment_vectors()  # fetch vibration table and vectors
            freqs = [*map(float, self.modes['Frequencies'])]  # extract frequencies
            if [*filter(lambda x: x < 0., freqs)]:  # check for negative frequencies
                print(self.log_file_path)
                raise NegativeFrequencyException()
        except TypeError:  # no frequencies
            print(self.log_file_path)
            raise OptimizationIncompleteException()

    def get_descriptors(self) -> dict:
        """Extract and retrieve all descriptors as a dictionary.

        :return: dict
        """

        self._extract_descriptors()
        # concatenate atom_desciptors from various sources
        self.atom_descriptors = pd.concat([self.geom[list('XYZ')],
                                           self.vbur,
                                           self.atom_freq_descriptors,
                                           self.atom_td_descriptors], axis=1)

        keys_to_save = ['labels', 'descriptors', 'atom_descriptors', 'transitions', 'modes', 'mode_vectors', ]
        dictionary = {key: value for key, value in self.__dict__.items() if key in keys_to_save}
        # convert dataframes to dicts
        for key, value in dictionary.items():
            if isinstance(value, pd.DataFrame):
                dictionary[key] = value.to_dict(orient='list')

        return dictionary

    def _extract_descriptors(self) -> None:
        """Extract all descriptor presets: buried volumes, vibrational modes, freq part descriptors and \
        and td part descriptors"""

        logger.debug(f"Extracting descriptors.")
        self.get_atom_labels()  # atom labels
        self.get_geometry()  # geometry
        self._compute_occupied_volumes()  # compute buried volumes
        self._get_frequencies_and_moment_vectors()
        self._get_freq_part_descriptors()  # fetch descriptors from frequency section
        self._get_td_part_descriptors()  # fetch descriptors from TD section

    def get_atom_labels(self) -> None:
        """Fetch the atom labels and store as attribute 'labels'."""

        # regex logic, fetch part between "Multiplicity =\d\n" and a double line
        # break (empty line may contain spaces)
        z_matrix = re.findall("Multiplicity = \d\n(.*?)\n\s*\n", self.log, re.DOTALL)[0]
        z_matrix = list(map(str.strip, z_matrix.split("\n")))
        # clean up extra lines if present
        if z_matrix[0].lower().startswith(('redundant', 'symbolic')):
            z_matrix = z_matrix[1:]
        if z_matrix[-1].lower().startswith('recover'):
            z_matrix = z_matrix[:-1]

        # fetch labels checking either space or comma split
        self.labels = []
        for line in z_matrix:
            space_split = line.split()
            comma_split = line.split(",")
            if len(space_split) > 1:
                self.labels.append(space_split[0])
            elif len(comma_split) > 1:
                self.labels.append(comma_split[0])
            else:
                raise Exception("Cannot fetch labels from geometry block")

    def get_geometry(self) -> None:
        """Extract geometry dataframe and store as attribute 'geom'."""

        # regex logic: find parts between "Standard orientation.*X Y Z" and "Rotational constants"
        geoms = re.findall("Standard orientation:.*?X\s+Y\s+Z\n(.*?)\n\s*Rotational constants",
                           self.log, re.DOTALL)

        # use the last available geometry block
        geom = geoms[-1]
        geom = map(str.strip, geom.splitlines())  # split lines and strip outer spaces
        geom = filter(lambda line: set(line) != {'-'}, geom)  # remove lines that only contain "---"
        geom = map(str.split, geom)  # split each line by space

        # convert to np.array for further manipulation (note: arrays have unique dtype, here it's str)
        geom_arr = np.array(list(geom))
        # create a dataframe
        geom_df = pd.concat([
            pd.DataFrame(geom_arr[:, 1:3].astype(int), columns=['AN', 'Type']),
            pd.DataFrame(geom_arr[:, 3:].astype(float), columns=list('XYZ'))
        ], axis=1)

        self.geom = geom_df

    def _compute_occupied_volumes(self, radius=3) -> None:
        """Calculate occupied volumes for each atom in the molecule."""

        logger.debug(f"Computing buried volumes within radius: {radius} Angstroms.")
        self.vbur = pd.Series(self.geom.index.map(lambda i: occupied_volume(self.geom, i, radius)),
                              name='VBur')

    def _split_parts(self) -> None:
        """Split the log file into parts that correspond to gaussian tasks."""

        # regex logic: log parts start with a new line and " # " pattern
        log_parts = re.split("\n\s-+\n\s#\s", self.log)[1:]  # TODO: possible ad-hoc behaviour
        self.parts = {}
        for p in log_parts:
            # regex logic: find first word in the text
            # name = re.search("^\w+", p).group(0).lower()
            # self.parts[name] = p
            names = re.search("^(\w|\s)+\s", p).group(0).lower().split(' ')[:-1]
            for n in names:
                self.parts[n] = p

    def _get_frequencies_and_moment_vectors(self) -> None:
        """Extract the vibrational modes and their moment vectors."""

        logger.debug("Extracting vibrational frequencies and moment vectors.")
        if 'freq' not in self.parts:
            logger.debug("Output file does not have a 'freq' part. Cannot extract frequencies.")
            return

        try:
            # regex logic: text between "Harmonic... normal coordinates and Thermochemistry, preceeded by a line of "---"
            freq_part = re.findall("Harmonic frequencies.*normal coordinates:\s*(\n.*?)\n\n\s-+\n.*Thermochemistry",
                                   self.parts['freq'], re.DOTALL)[0]

            # split each section of text with frequencies
            # regex logic, each frequency part ends with a \s\d+\n, note: we do not use DOTALL here!
            freq_sections = re.split("\n.*?\s\d+\n", freq_part)[1:]

            freq_dfs, vector_dfs = [], []
            for freq_section in freq_sections:
                # frequencies
                freqs = re.findall("\n(\s\w+.*?)\n\s+Atom", freq_section, re.DOTALL)[0]
                freqs = [text.split("--") for text in freqs.splitlines()]
                freqs = [[item[0].strip()] + item[1].split() for item in freqs]
                freqs = np.array(freqs).T.tolist()
                freq_dfs.append(pd.DataFrame(freqs[1:], columns=[name.replace(".", "") for name in freqs[0]]))

                # vectors
                vectors = re.findall("\n(\s+Atom.*)", freq_section, re.DOTALL)[0]
                vectors = [text.split() for text in vectors.splitlines()]
                vector_df = pd.DataFrame(vectors[1:], columns=vectors[0])
                vector_df.drop(["Atom", "AN"], axis=1, inplace=True)
                vector_dfs.append(vector_df)

            # combine into one frame
            frequencies = pd.concat(freq_dfs)
            frequencies['mode_number'] = range(1, len(frequencies) + 1)
            self.modes = frequencies.set_index('mode_number').astype(float)

            vectors = pd.concat(vector_dfs, axis=1).astype(float)
            vectors.columns = pd.MultiIndex.from_product([list(range(1, len(frequencies) + 1)), ['X', 'Y', 'Z'], ])
            vectors.columns.names = ['mode_number', 'axis']
            vectors = vectors.unstack().reset_index(level=2, drop=True).to_frame('value').reset_index()
            self.mode_vectors = vectors
        except Exception:
            self.modes = None
            self.mode_vectors = None
            logger.warning("Log file does not contain vibrational frequencies")

    def _get_freq_part_descriptors(self) -> None:
        """Extract descriptors from frequency part."""

        logger.debug("Extracting frequency section descriptors")
        if 'freq' not in self.parts:
            logger.info("Output file does not have a 'freq' section. Cannot extract descriptors.")
            return

        text = self.parts['freq']

        # single value descriptors
        single_value_desc_list = [
            {"name": "number_of_atoms", "prefix": "NAtoms=\s*", "type": int},
            {"name": "charge", "prefix": "Charge\s=\s*", "type": int},
            {"name": "multiplicity", "prefix": "Multiplicity\s=\s*", "type": int},
            {"name": "dipole", "prefix": "Dipole moment \(field-independent basis, Debye\):.*?Tot=\s*", "type": float},
            {"name": "molar_mass", "prefix": "Molar Mass =\s*", "type": float},
            {"name": "molar_volume", "prefix": "Molar volume =\s*", "type": float},
            {"name": "electronic_spatial_extent", "prefix": "Electronic spatial extent\s+\(au\):\s+<R\*\*2>=\s*",
             "type": float},
            {"name": "E_scf", "prefix": "SCF Done:\s+E.*?=\s*", "type": float},
            {"name": "zero_point_correction", "prefix": "Zero-point correction=\s*", "type": float},
            {"name": "E_thermal_correction", "prefix": "Thermal correction to Energy=\s*", "type": float},
            {"name": "H_thermal_correction", "prefix": "Thermal correction to Enthalpy=\s*", "type": float},
            {"name": "G_thermal_correction", "prefix": "Thermal correction to Gibbs Free Energy=\s*", "type": float},
            {"name": "E_zpe", "prefix": "Sum of electronic and zero-point Energies=\s*", "type": float},
            {"name": "E", "prefix": "Sum of electronic and thermal Energies=\s*", "type": float},
            {"name": "H", "prefix": "Sum of electronic and thermal Enthalpies=\s*", "type": float},
            {"name": "G", "prefix": "Sum of electronic and thermal Free Energies=\s*", "type": float},
        ]

        for desc in single_value_desc_list:
            for part_name in ['freq', 'opt']:
                try:
                    value = re.search(f"{desc['prefix']}({float_or_int_regex})",
                                      self.parts[part_name],
                                      re.DOTALL).group(1)
                    self.descriptors[desc["name"]] = desc['type'](value)
                except (AttributeError, KeyError):
                    pass
            if desc["name"] not in self.descriptors:
                self.descriptors[desc["name"]] = None
                logger.debug(f'''Descriptor {desc["name"]} not present in the log file.''')

        # stoichiometry
        self.descriptors['stoichiometry'] = re.search("Stoichiometry\s*(\w+)", text).group(1)

        # convergence, regex-logic: last word in each line should be "YES"
        try:
            string = re.search("(Maximum Force.*?)\sPredicted change", text, re.DOTALL).group(1)
            conv = list(map(lambda x: x[0], re.findall("(\w+)\s?(\r\n|\r|\n)", string)))

            # compute the fraction of YES/NO answers
            self.descriptors['converged'] = (np.array(conv) == 'YES').mean()  # Before it was "(\w+)\n"
        except Exception:
            self.descriptors['converged'] = None
            logger.warning("Log file does not have optimization convergence information")

        # energies, regex-logic: find all floats in energy block, split by occupied, virtual orbitals
        string = re.search("Population.*?SCF [Dd]ensity.*?(\sAlph.*?)\n\s*Condensed", text, re.DOTALL).group(1)
        if self.descriptors['multiplicity'] == 1:
            energies = [re.findall(f"({float_or_int_regex})", s_part) for s_part in string.split("Alpha virt.", 1)]
            occupied_energies, unoccupied_energies = [map(float, e) for e in energies]
            homo, lumo = max(occupied_energies), min(unoccupied_energies)
        elif self.descriptors['multiplicity'] == 3:
            alpha, beta = re.search("(\s+Alpha\s+occ. .*?)(\s+Beta\s+occ. .*)", string, re.DOTALL).groups()
            energies_alpha = [re.findall(f"({float_or_int_regex})", s_part) for s_part in alpha.split("Alpha virt.", 1)]
            energies_beta = [re.findall(f"({float_or_int_regex})", s_part) for s_part in beta.split("Beta virt.", 1)]
            occupied_energies_alpha, unoccupied_energies_alpha = [map(float, e) for e in energies_alpha]
            occupied_energies_beta, unoccupied_energies_beta = [map(float, e) for e in energies_beta]
            homo_alpha, lumo_alpha = max(occupied_energies_alpha), min(unoccupied_energies_alpha)
            homo_beta, lumo_beta = max(occupied_energies_beta), min(unoccupied_energies_beta)
            homo, lumo = homo_alpha, lumo_beta
        else:
            logger.warning(f"Unsupported multiplicity {self.descriptors['multiplicity']}, cannot compute homo/lumo. "
                           f"Setting both to 0.")
            homo, lumo = 0, 0
        self.descriptors['homo_energy'] = homo
        self.descriptors['lumo_energy'] = lumo
        self.descriptors['electronegativity'] = -0.5 * (lumo + homo)
        self.descriptors['hardness'] = 0.5 * (lumo - homo)

        # atom_dependent section
        # Mulliken population
        string = re.search("Mulliken charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
        charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        if len(charges) < len(self.labels):
            string = re.search("Mulliken atomic charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
            charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        mulliken = pd.Series(charges, name='Mulliken_charge')

        # APT charges
        try:
            string = re.search("APT charges.*?\n(.*?)\n\s*Sum of APT", text, re.DOTALL).group(1)
            charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
            apt = pd.Series(charges, name='APT_charge')
        except (IndexError, AttributeError):
            try:
                string = re.search("APT atomic charges.*?\n(.*?)\n\s*Sum of APT", text, re.DOTALL).group(1)
                charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
                apt = pd.Series(charges, name='APT_charge')
            except Exception:
                apt = pd.Series(name='APT_charge')
                logger.warning(f"Log file does not contain APT charges.")

        # NPA charges
        try:
            string = re.search("Summary of Natural Population Analysis:.*?\n\s-+\n(.*?)\n\s=+\n", text,
                               re.DOTALL).group(1)
            population = np.array(list(map(str.split, string.splitlines())))[:, 2:]
            npa = pd.DataFrame(population,
                               columns=['NPA_charge', 'NPA_core', 'NPA_valence', 'NPA_Rydberg', 'NPA_total'])
        except Exception:
            npa = pd.DataFrame(['NPA_charge', 'NPA_core', 'NPA_valence', 'NPA_Rydberg', 'NPA_total'])
            logger.debug(f"Log file does not contain NPA charges.")

        # NMR
        try:
            string = re.findall(f"Isotropic\s=\s*({float_or_int_regex})\s*Anisotropy\s=\s*({float_or_int_regex})", text)
            nmr = pd.DataFrame(np.array(string).astype(float), columns=['NMR_shift', 'NMR_anisotropy'])
        except Exception:
            nmr = pd.DataFrame(columns=['NMR_shift', 'NMR_anisotropy'])
            logger.debug(f"Log file does not contain NMR shifts.")

        self.atom_freq_descriptors = pd.concat([mulliken, apt, npa, nmr], axis=1)

    def _get_td_part_descriptors(self) -> None:
        """Extract descriptors from TD part."""

        logger.debug("Extracting TD section descriptors")
        if 'td' not in self.parts:
            logger.debug("Output file does not have a 'TD' section. Cannot extract descriptors.")
            return

        text = self.parts['td']

        single_value_desc_list = [
            {"name": "ES_root_dipole", "prefix": "Dipole moment \(field-.*?, Debye\):.*?Tot=\s*", "type": float},
            {"name": "ES_root_molar_volume", "prefix": "Molar volume =\s*", "type": float},
            {"name": "ES_root_electronic_spatial_extent",
             "prefix": "Electronic spatial extent\s+\(au\):\s+<R\*\*2>=\s*", "type": float},
        ]

        for desc in single_value_desc_list:
            value = re.search(f"{desc['prefix']}({float_or_int_regex})", text, re.DOTALL).group(1)
            self.descriptors[desc["name"]] = desc['type'](value)

        # excited states
        string = re.findall(f"Excited State.*?({float_or_int_regex})\snm"
                            f".*f=({float_or_int_regex})"
                            f".*<S\*\*2>=({float_or_int_regex})", text)
        self.transitions = pd.DataFrame(np.array(string).astype(float),
                                        columns=['ES_transition', 'ES_osc_strength', 'ES_<S**2>'])

        # atom_dependent section
        # Mulliken population
        string = re.search("Mulliken charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
        charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        mulliken = pd.Series(charges, name='ES_root_Mulliken_charge')

        # NPA charges
        string = re.search("Summary of Natural Population Analysis:.*?\n\s-+\n(.*?)\n\s=+\n", text, re.DOTALL).group(1)
        population = np.array(list(map(str.split, string.splitlines())))[:, 2:]
        npa = pd.DataFrame(population, columns=['ES_root_NPA_charge', 'ES_root_NPA_core', 'ES_root_NPA_valence',
                                                'ES_root_NPA_Rydberg', 'ES_root_NPA_total'])

        self.atom_td_descriptors = pd.concat([mulliken, npa], axis=1)


class NegativeFrequencyException(Exception):
    """Raised when a negative frequency is found in the Gaussian log file. The geometry did not converge,
    and the job shall be resubmitted."""
    pass


class NoGeometryException(Exception):
    """Raised when Gaussian does not contain geometry information. Job failed early and cannot be fixed by
    resubmission."""
    pass


class OptimizationIncompleteException(Exception):
    """Raised when the optimization has not completed successfully."""
    pass


def test_extractor(log_path):
    extractor = gaussian_log_extractor(log_path)
    desc = extractor.get_descriptors()
    print(desc)


def unroll_list(dictionary, key):
    """
    Transform a dictionary key that is a list into a pandas dataframe with one row where each value is an
    independent row. For example, a key named 'X' whose value is [1,2,3,4] will be transformed into this dataframe:
        X_0       X_1       X_2       X_3
    0    1         2         3         4
    """
    col_names = [str(key) + f'_{i}' for i in range(len(dictionary[key]))]

    return pd.DataFrame.from_records([dictionary[key]], columns=col_names)


def extract_df_from_part(dictionary):
    """
    Extract a dataframe from a part in the descriptor dictionary from the gaussian_log_extractor
    """
    res = pd.DataFrame()

    if isinstance(dictionary, dict):
        for k in dictionary.keys():
            if isinstance(dictionary[k], list):
                df = unroll_list(dictionary, k)
                res = pd.concat([res, df], axis=1)
            else:
                res.loc[0, k] = dictionary[k]
    else:
        logger.warning('Possible error in the .log file. Check for empty columns in the .csv file.')

    return res


def dict_csv_conversor(dictionary):
    """
    Convert the output from a gaussian_log_extractor into a pandas dataframe to export it into a .csv
    """
    res = pd.DataFrame()

    for k in dictionary.keys():
        df = extract_df_from_part(dictionary[k])
        res = pd.concat([res, df], axis=1)

    return res


def filter_extractor_dict(dictionary):
    if dictionary['descriptors']:
        dictionary['descriptors'].pop('molar_mass', None)  # Delete for now
        dictionary['descriptors'].pop('molar_volume', None)
        dictionary['descriptors'].pop('converged', None)
    if dictionary['atom_descriptors']:
        dictionary['atom_descriptors'].pop('X', None)
        dictionary['atom_descriptors'].pop('Y', None)
        dictionary['atom_descriptors'].pop('Z', None)
        dictionary['atom_descriptors'].pop('VBur', None)
        dictionary['atom_descriptors'].pop('Mulliken_charge', None)
        dictionary['atom_descriptors'].pop('APT_charge', None)
        dictionary['atom_descriptors'].pop(0, None)
        dictionary['atom_descriptors'].pop('NMR_shift', None)
        dictionary['atom_descriptors'].pop('NMR_anisotropy', None)
        dictionary['atom_descriptors'].pop('NPA_core', None)

    if dictionary['modes']:
        dictionary['modes'].pop('Frequencies', None)
        dictionary['modes'].pop('Red masses', None)
        dictionary['modes'].pop('Frc consts', None)
        dictionary['modes'].pop('IR Inten', None)
    dictionary.pop('mode_vectors', None)
    dictionary.pop('transitions', None)
    # dictionary['labels'] = {'labels': dictionary['labels']}  # Labels gives an error
    dictionary.pop('labels', None)


def export_to_pandas(log_path):
    extractor = gaussian_log_extractor(log_path)
    desc = extractor.get_descriptors()
    filter_extractor_dict(desc)
    df = dict_csv_conversor(desc)

    return df


def export_to_csv(dir_path):
    """ Loads all files in the dir_path folder and subdirectories and processes each .log file found """
    dir = os.fsencode(dir_path)
    res = pd.DataFrame()

    for file in tqdm(os.listdir(dir)):
        filename = os.fsdecode(file)
        if filename.endswith(".log"):
            path = os.path.join(dir_path, filename)
            df = export_to_pandas(path)
            comp_name = filename.split('.')[0]
            df.insert(0, 'Name', [comp_name])
            if len(df.columns) > len(res.columns):
                res = pd.concat([df, res], axis=0)
            else:
                res = pd.concat([res, df], axis=0)

    res.reset_index(drop=True, inplace=True)
    res.to_csv('extracted_values.csv', index=False)


def compute_log_files(data_dir='./'):
    """
    Recursive function that join the dft properties inside separate .log files into a single one
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
        logger.info(f'Now processing file {n}')
        try:
            df = export_to_pandas(f'{data_dir}/{n}.log')
            df.insert(0, 'file_name', n)  # Insert the file name for joining with morfeus output later
            if len(df.columns) > len(res.columns):
                res = pd.concat([df, res], axis=0)
            else:
                res = pd.concat([res, df], axis=0)
        except AttributeError as e:
            logging.warning(f'Possible problem processing the log file: {e}')

    # Recursive case, further directories
    if len(dirs[1]) > 0:
        prev_dir = os.getcwd()
        for d in dirs[1]:
            # Change the wd to the subdirectory
            res_rec = compute_log_files(f'{prev_dir}/{d}')
            if len(res_rec.columns) > len(res.columns):
                res = pd.concat([res_rec, res], axis=0)
            else:
                res = pd.concat([res, res_rec], axis=0)

        os.chdir(prev_dir)

    return res


if __name__ == "__main__":
    # Prepare the logger to output into both console and a file with the desired format
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description='Generate dft .csv dataset from intermediary .log files')
    parser.add_argument("--data_dir", type=str, help="Directory where to search for .log files", default='./')

    args = parser.parse_args()

    logger.info('Started')

    res = compute_log_files(data_dir=args.data_dir)
    res.reset_index(drop=True, inplace=True)
    res.to_csv('log_values.csv', index=False)

    logger.info('Finished')
