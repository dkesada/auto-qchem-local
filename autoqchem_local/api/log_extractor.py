import logging
import pandas as pd

from autoqchem_local.autoqchem.gaussian_log_extractor import gaussian_log_extractor

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
            dictionary['atom_descriptors'].pop('NMR_shift', None)
            dictionary['atom_descriptors'].pop('NMR_anisotropy', None)

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

    def export_to_pandas(self, log_path):
        """
        Main access point to the log extractors. Converts a .log file into a pandas dataframe and returns it
        :param log_path: the path to the .log file
        :return: a pandas dataframe with the extracted info from the .log file
        """
        extractor = gaussian_log_extractor(log_path, self.logger)
        desc = extractor.get_descriptors()
        self.filter_extractor_dict(desc)
        df = self.dict_csv_conversor(desc)

        return df
