import logging
import pandas as pd
import numpy as np

from autoqchem_local.autoqchem.gaussian_log_extractor import gaussian_log_extractor
from autoqchem_local.morfeus_ml.geometry import (convert_to_symbol, get_first_idx, get_all_closest_atom_to_metal,
                                                 get_closest_atom_to_metal, get_three_point_angle, euclid_dist)

try:
    from openbabel import pybel  # openbabel 3.0.0

    GetVdwRad = pybel.ob.GetVdwRad
except ImportError:
    import pybel  # openbabel 2.4

    table = pybel.ob.OBElementTable()
    GetVdwRad = table.GetVdwRad


logger = logging.getLogger(__name__)


class LogExtractor:
    def __init__(self, log=None):
        if log:
            self.logger = log
        else:
            self.logger = logger

    @staticmethod
    def unroll_list(dictionary, key):
        """
        Transform a dictionary key that is a list into a pandas dataframe with one row where each value is an
        independent row. For example, a key named 'X' whose value is [1,2,3,4] will be transformed into this dataframe:
            X_0       X_1       X_2       X_3
        0    1         2         3         4
        """
        col_names = [str(key) + f'_{i}' for i in range(len(dictionary[key]))]

        return pd.DataFrame.from_records([dictionary[key]], columns=col_names)

    def extract_df_from_part(self, dictionary):
        """
        Extract a dataframe from a part in the descriptor dictionary from the gaussian_log_extractor
        """
        res = pd.DataFrame()

        if isinstance(dictionary, dict):
            for k in dictionary.keys():
                if isinstance(dictionary[k], list):
                    df = self.unroll_list(dictionary, k)
                    res = pd.concat([res, df], axis=1)
                else:
                    res.loc[0, k] = dictionary[k]
        else:
            logger.warning('Possible error in the .log file. Check for empty columns in the .csv file.')

        return res

    def dict_csv_conversor(self, dictionary):
        """
        Convert the output from a gaussian_log_extractor into a pandas dataframe to export it into a .csv
        """
        res = pd.DataFrame()

        for k in dictionary.keys():
            df = self.extract_df_from_part(dictionary[k])
            res = pd.concat([res, df], axis=1)

        return res


    @staticmethod
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
            dictionary['atom_descriptors'].pop('NPA_charge', None)
            dictionary['atom_descriptors'].pop('NPA_core', None)
            dictionary['atom_descriptors'].pop('NPA_valence', None)
            dictionary['atom_descriptors'].pop('NPA_Rydberg', None)
            dictionary['atom_descriptors'].pop('NPA_total', None)

        if dictionary['modes']:
            dictionary['modes'].pop('Frequencies', None)
            dictionary['modes'].pop('Red masses', None)
            dictionary['modes'].pop('Frc consts', None)
            dictionary['modes'].pop('IR Inten', None)
            dictionary['modes'].pop('Dip str', None)
            dictionary['modes'].pop('Rot str', None)
            dictionary['modes'].pop('E-M angle', None)

        dictionary.pop('mode_vectors', None)
        dictionary.pop('transitions', None)
        # dictionary['labels'] = {'labels': dictionary['labels']}  # Labels gives an error
        dictionary.pop('labels', None)

    @staticmethod
    def get_elems_and_coords(extractor, desc):
        # Obtain the element list and coordinates
        coordinates = np.array([[x, y, z] for x, y, z in
                                zip(desc['atom_descriptors']['X'], desc['atom_descriptors']['Y'],
                                    desc['atom_descriptors']['Z'])])

        try:
            elements = convert_to_symbol([int(e) for e in extractor.labels])
        except ValueError:
            elements = extractor.labels

        return elements, coordinates

    @staticmethod
    def calculate_pd_angle_dist(elements, coordinates):
        """
        Calculate the angles and distances in the cross P-Pd-P Cl-Pd-Cl. One of the P can also be a C in some cases.
        Returns all these values inside a dictionary
        """

        # Get the necessary indices of all involved elements
        pd_idx = get_first_idx('Pd', elements)
        p_idx = get_all_closest_atom_to_metal('P', elements, pd_idx, coordinates)
        cl_idx = get_all_closest_atom_to_metal('Cl', elements, pd_idx, coordinates)[0:2]  # Closest two Cl to Pd
        if len(p_idx) < 2:
            p_idx = np.append(p_idx, get_closest_atom_to_metal('C', elements, pd_idx, coordinates))  # Closest C if no double P

        res = {}

        # Calculate the angles
        res['p_pd_p_angle'] = get_three_point_angle(coordinates[pd_idx], coordinates[p_idx[0]], coordinates[p_idx[1]])
        res['cl_pd_cl_angle'] = get_three_point_angle(coordinates[pd_idx], coordinates[cl_idx[0]],
                                                      coordinates[cl_idx[1]])

        # Calculate the distances, from closest to furthest. ppd1 will always be the closest P (or C) to the Pd
        res['ppd1'] = euclid_dist(coordinates[pd_idx], coordinates[p_idx[0]])
        res['ppd2'] = euclid_dist(coordinates[pd_idx], coordinates[p_idx[1]])
        res['clpd1'] = euclid_dist(coordinates[pd_idx], coordinates[cl_idx[0]])
        res['clpd2'] = euclid_dist(coordinates[pd_idx], coordinates[cl_idx[1]])

        return res

    @staticmethod
    def extract_pd_props(elements, coordinates, desc):
        """
        Extract some specific useful properties for paladium
        """
        # Get the necessary indices of all involved elements
        pd_idx = get_first_idx('Pd', elements)
        p_idx = get_all_closest_atom_to_metal('P', elements, pd_idx, coordinates)
        cl_idx = get_all_closest_atom_to_metal('Cl', elements, pd_idx, coordinates)[0:2]  # Closest two Cl to Pd
        if len(p_idx) < 2:
            p_idx = np.append(p_idx, get_closest_atom_to_metal('C', elements, pd_idx, coordinates))  # Closest C if no double P

        res = {}

        # APT charges
        res['p1_apt_charge'] = float(desc['atom_descriptors']['APT_charge'][p_idx[0]])
        res['p2_apt_charge'] = float(desc['atom_descriptors']['APT_charge'][p_idx[1]])
        res['pd_atp_charge'] = float(desc['atom_descriptors']['APT_charge'][pd_idx])
        res['cl1_apt_charge'] = float(desc['atom_descriptors']['APT_charge'][cl_idx[0]])
        res['cl2_apt_charge'] = float(desc['atom_descriptors']['APT_charge'][cl_idx[1]])

        # NMR
        res['p1_nmr_shift'] = float(desc['atom_descriptors']['NMR_shift'][p_idx[0]])
        res['p2_nmr_shif'] = float(desc['atom_descriptors']['NMR_shift'][p_idx[1]])
        res['p1_nmr_anis'] = float(desc['atom_descriptors']['NMR_anisotropy'][p_idx[0]])
        res['p2_nmr_anis'] = float(desc['atom_descriptors']['NMR_anisotropy'][p_idx[1]])

        # NPA charge
        res['p1_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][p_idx[0]])
        res['p2_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][p_idx[1]])
        res['pd_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][pd_idx])
        res['cl1_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][cl_idx[0]])
        res['cl2_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][cl_idx[1]])

        # NPA core
        res['p1_npa_core'] = float(desc['atom_descriptors']['NPA_core'][p_idx[0]])
        res['p2_npa_core'] = float(desc['atom_descriptors']['NPA_core'][p_idx[1]])
        res['pd_npa_core'] = float(desc['atom_descriptors']['NPA_core'][pd_idx])
        res['cl1_npa_core'] = float(desc['atom_descriptors']['NPA_core'][cl_idx[0]])
        res['cl2_npa_core'] = float(desc['atom_descriptors']['NPA_core'][cl_idx[1]])

        # NPA valence
        res['p1_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][p_idx[0]])
        res['p2_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][p_idx[1]])
        res['pd_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][pd_idx])
        res['cl1_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][cl_idx[0]])
        res['cl2_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][cl_idx[1]])

        # NPA Rydberg
        res['p1_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][p_idx[0]])
        res['p2_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][p_idx[1]])
        res['pd_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][pd_idx])
        res['cl1_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][cl_idx[0]])
        res['cl2_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][cl_idx[1]])

        # NPA total
        res['p1_npa_total'] = float(desc['atom_descriptors']['NPA_total'][p_idx[0]])
        res['p2_npa_total'] = float(desc['atom_descriptors']['NPA_total'][p_idx[1]])
        res['pd_npa_total'] = float(desc['atom_descriptors']['NPA_total'][pd_idx])
        res['cl1_npa_total'] = float(desc['atom_descriptors']['NPA_total'][cl_idx[0]])
        res['cl2_npa_total'] = float(desc['atom_descriptors']['NPA_total'][cl_idx[1]])

        return res

    @staticmethod
    def extract_mo_props(elements, coordinates, desc):
        """
        Extract some specific useful properties for molybdenum
        """
        # Get the necessary indices of all involved elements
        mo_idx = get_first_idx('Mo', elements)

        res = {}

        # Mulliken charge
        res['mo_mulliken_charge'] = float(desc['atom_descriptors']['Mulliken_charge'][mo_idx])

        # APT charges
        res['mo_apt_charge'] = float(desc['atom_descriptors']['APT_charge'][mo_idx])

        # NMR shift
        res['mo_nmr_shift'] = float(desc['atom_descriptors']['NMR_shift'][mo_idx])

        # NPA charge
        res['mo_npa_charge'] = float(desc['atom_descriptors']['NPA_charge'][mo_idx])

        # NPA valence
        res['mo_npa_valence'] = float(desc['atom_descriptors']['NPA_valence'][mo_idx])

        # NPA Rydberg
        res['mo_npa_rydberg'] = float(desc['atom_descriptors']['NPA_Rydberg'][mo_idx])

        # NPA total
        res['mo_npa_total'] = float(desc['atom_descriptors']['NPA_total'][mo_idx])

        return res

    @staticmethod
    def extract_closer_o_props(elements, coordinates, desc):
        """
        Extract some specific useful properties for molybdenum
        """
        # Get the necessary indices of all involved elements
        mo_idx = get_first_idx('Mo', elements)
        o_idx = get_all_closest_atom_to_metal('O', elements, mo_idx, coordinates)  # All oxygen atoms by proximity

        res = {}

        # Mulliken charge
        res['o_mulliken_charge'] = [float(i) for i in np.array(desc['atom_descriptors']['Mulliken_charge'])[o_idx]]

        # APT charges
        res['o_apt_charge'] = [float(i) for i in np.array(desc['atom_descriptors']['APT_charge'])[o_idx]]

        # NMR shift
        res['o_nmr_shift'] = [float(i) for i in np.array(desc['atom_descriptors']['NMR_shift'])[o_idx]]

        # NPA charge
        res['o_npa_charge'] = [float(i) for i in np.array(desc['atom_descriptors']['NPA_charge'])[o_idx]]

        # NPA valence
        res['o_npa_valence'] = [float(i) for i in np.array(desc['atom_descriptors']['NPA_valence'])[o_idx]]

        # NPA Rydberg
        res['o_npa_rydberg'] = [float(i) for i in np.array(desc['atom_descriptors']['NPA_Rydberg'])[o_idx]]

        # NPA total
        res['o_npa_total'] = [float(i) for i in np.array(desc['atom_descriptors']['NPA_total'])[o_idx]]

        return res

    def export_to_pandas(self, log_path):
        """
        Main access point to the log extractors. Converts a .log file into a pandas dataframe and returns it
        :param log_path: the path to the .log file
        :return: a pandas dataframe with the extracted info from the .log file
        """
        extractor = gaussian_log_extractor(log_path, self.logger)
        desc = extractor.get_descriptors()

        # Specific calculations, usually commented
        #elements, coordinates = self.get_elems_and_coords(extractor, desc)
        #desc['pd_angle_dist'] = self.calculate_pd_angle_dist(elements, coordinates)
        #desc['pd_vals'] = self.extract_pd_props(elements, coordinates, desc)
        #desc['mo_vals'] = self.extract_mo_props(elements, coordinates, desc)
        #desc['o_vals'] = self.extract_closer_o_props(elements, coordinates, desc)

        self.filter_extractor_dict(desc)
        df = self.dict_csv_conversor(desc)

        return df
