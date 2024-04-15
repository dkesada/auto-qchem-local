import logging
import re
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from autoqchem.descriptor_functions import occupied_volume
from autoqchem.gaussian_log_extractor import gaussian_log_extractor, NegativeFrequencyException, NoGeometryException, OptimizationIncompleteException

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

        self.float_or_int_regex = "[-+]?[0-9]*\.[0-9]+|[0-9]+"
